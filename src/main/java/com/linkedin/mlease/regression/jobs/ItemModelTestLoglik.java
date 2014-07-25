/**
 * Copyright 2014 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.linkedin.mlease.regression.jobs;

import java.io.IOException;
import java.util.Map;

import org.apache.avro.Schema;
import org.apache.avro.Schema.Type;
import org.apache.avro.generic.GenericData;
import org.apache.avro.mapred.AvroCollector;
import org.apache.avro.mapred.AvroMapper;
import org.apache.avro.mapred.AvroReducer;
import org.apache.avro.mapred.Pair;
import org.apache.avro.util.Utf8;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;
import org.apache.log4j.Logger;

import com.linkedin.mlease.regression.avro.RegressionTestLoglikOutput;
import com.linkedin.mlease.utils.Util;
import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.AvroUtils;
import com.linkedin.mapred.JobConfig;


public class ItemModelTestLoglik extends AbstractAvroJob
{
  public static final Logger _logger = Logger.getLogger(ItemModelTestLoglik.class);
  
  public ItemModelTestLoglik(String name, JobConfig config)
  {
    super(name, config);
  }
  @Override
  public void run() throws Exception
  {
    JobConfig props = super.getJobConfig();
    JobConf conf = super.createJobConf(ItemModelTestLoglikMapper.class,
                                       ItemModelTestLoglikReducer.class,
                                       ItemModelTestLoglikCombiner.class,
                                       Pair.getPairSchema(Schema.create(Type.STRING),
                                                          RegressionTestLoglikOutput.SCHEMA$),
                                                          RegressionTestLoglikOutput.SCHEMA$);
    AvroUtils.runAvroJob(conf);
  }
  public static class ItemModelTestLoglikMapper extends AvroMapper<GenericData.Record, Pair<String, RegressionTestLoglikOutput>>
  {
    @Override
    public void map(GenericData.Record data,
                    AvroCollector<Pair<String, RegressionTestLoglikOutput>> collector,
                    Reporter reporter) throws IOException
    {
      int response = Util.getIntAvro(data, "response");
      Map<Utf8, Float> pred = (Map<Utf8, Float>) data.get("pred");
      double weight = 1;
      if (data.get("weight")!=null)
      {
        weight = Util.getDoubleAvro(data, "weight");
      }
      if (response != 1 && response != 0 && response != -1)
      {
        throw new IOException("response should be 1,0 or -1!");
      }
      for (Utf8 k : pred.keySet())
      {
        double loglik = 0;
        if (response == 1)
        {
          loglik = -Math.log1p(Math.exp(-pred.get(k))) * weight;
        }
        else
        {
          loglik = -Math.log1p(Math.exp(pred.get(k))) * weight;
        }
        RegressionTestLoglikOutput output = new RegressionTestLoglikOutput();
        output.key = k;
        output.testLoglik = (float) loglik;
        output.count = weight;
        collector.collect(new Pair<String, RegressionTestLoglikOutput>(k.toString(), output));
      }
    }
  }
  public static class ItemModelTestLoglikReducer extends AvroReducer<Utf8, RegressionTestLoglikOutput, RegressionTestLoglikOutput>
  {
    @Override
    public void reduce(Utf8 key,
                       Iterable<RegressionTestLoglikOutput> values,
                       AvroCollector<RegressionTestLoglikOutput> collector,
                       Reporter reporter) throws IOException
    {
      double sumLoglik = 0;
      double n = 0;
      for (RegressionTestLoglikOutput value : values)
      {
        float loglik = value.testLoglik;
        sumLoglik += loglik;
        n += value.count;
      }
      RegressionTestLoglikOutput output = new RegressionTestLoglikOutput();
      output.key = key;
      output.testLoglik = (float) (sumLoglik / n);
      output.count = n;
      collector.collect(output);
    }
  }
  public static class ItemModelTestLoglikCombiner extends AvroReducer<Utf8, RegressionTestLoglikOutput, Pair<Utf8, RegressionTestLoglikOutput>>
  {
    @Override
    public void reduce(Utf8 key,
                       Iterable<RegressionTestLoglikOutput> values,
                       AvroCollector<Pair<Utf8, RegressionTestLoglikOutput>> collector,
                       Reporter reporter) throws IOException
    {
      double sumLoglik = 0;
      double n = 0;
      for (RegressionTestLoglikOutput value : values)
      {
        float loglik = value.testLoglik;
        sumLoglik += loglik;
        n += value.count;
      }
      RegressionTestLoglikOutput output = new RegressionTestLoglikOutput();
      output.key = key;
      output.testLoglik = (float) sumLoglik;
      output.count = n;
      collector.collect(new Pair<Utf8, RegressionTestLoglikOutput>(key, output));
    }
  }
}