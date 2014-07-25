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
import java.util.HashSet;
import java.util.Set;

import org.apache.avro.mapred.AvroCollector;
import org.apache.avro.mapred.AvroMapper;
import org.apache.avro.mapred.AvroReducer;
import org.apache.avro.mapred.Pair;
import org.apache.avro.util.Utf8;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.Reporter;

import com.linkedin.mlease.regression.avro.RegressionPrepareOutput;

/**
 * The partitionId assigner for jobs like NaiveTrain
 * 
 * 
 */
public class PartitionIdAssigner
{
  public static final String LAMBDA = "lambda";

  public static class PartitionIdAssignerMapper extends
      AvroMapper<RegressionPrepareOutput, Pair<String, Integer>>
  {
    Set<Float> _lambdaSet;

    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _lambdaSet = new HashSet<Float>();
      String[] lambdastr = conf.get(LAMBDA).split(",");
      for (String l : lambdastr)
      {
        _lambdaSet.add(Float.parseFloat(l));
      }
    }

    @Override
    public void map(RegressionPrepareOutput data,
                    AvroCollector<Pair<String, Integer>> collector,
                    Reporter reporter) throws IOException
    {
      String key = data.key.toString();
      for (float lambda : _lambdaSet)
      {
        String newkey = String.valueOf(lambda) + "#" + key;
        data.key = newkey;
        Pair<String, Integer> outPair = new Pair<String, Integer>(newkey, 1);
        collector.collect(outPair);
      }
    }
  }
  public static class PartitionIdAssignerReducer extends
           AvroReducer<Utf8, Integer, Pair<String, Integer>>
  {
    int _partitionId = 0;
    @Override
    public void reduce(Utf8 key,
                       Iterable<Integer> values,
                       AvroCollector<Pair<String, Integer>> collector,
                       Reporter reporter) throws IOException
    {
      collector.collect(new Pair<String, Integer>(key, _partitionId));
      _partitionId++;
    }
  }
  public static class PartitionIdAssignerCombiner extends
           AvroReducer<Utf8, Integer, Pair<String, Integer>>
  {
    @Override
    public void reduce(Utf8 key,
                       Iterable<Integer> values,
                       AvroCollector<Pair<String, Integer>> collector,
                       Reporter reporter) throws IOException
    {
      collector.collect(new Pair<String, Integer>(key, 1));
    }
  }
}
