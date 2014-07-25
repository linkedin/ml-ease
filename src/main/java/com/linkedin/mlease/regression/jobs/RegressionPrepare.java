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
import java.util.ArrayList;
import java.util.List;

import org.apache.avro.generic.GenericData;
import org.apache.avro.mapred.AvroCollector;
import org.apache.avro.mapred.AvroMapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;
import org.apache.log4j.Logger;

import com.linkedin.mlease.regression.avro.RegressionPrepareOutput;
import com.linkedin.mlease.regression.avro.feature;
import com.linkedin.mlease.utils.Util;
import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.AvroUtils;
import com.linkedin.mapred.JobConfig;
/**
 * The preparation job for Regression, must run before running RegressionAdmmTrain or RegressionNaiveTrain etc.
 *
 */
public class RegressionPrepare extends AbstractAvroJob
{
  public static final Logger _logger              = Logger.getLogger(RegressionPrepare.class);
  public static final String MAP_KEY              = "map.key";
  public static final String NUM_BLOCKS           = "num.blocks";
  public static final String IGNORE_FEATURE_VALUE = "binary.feature";
  public static final String NUM_CLICK_REPLICATES = "num.click.replicates";
  
  public RegressionPrepare(String jobId, JobConfig config)
  {
    super(jobId, config);
  }
  public RegressionPrepare(JobConfig config)
  {
    super(config);
  }

  @Override
  public void run() throws Exception
  {
    JobConfig config = super.getJobConfig();
    JobConf conf =
        super.createJobConf(RegressionPrepareMapper.class,
                            RegressionPrepareOutput.SCHEMA$); 
    String mapKey = config.getString(MAP_KEY, "");
    conf.set(MAP_KEY, mapKey);
    conf.setInt(NUM_CLICK_REPLICATES, config.getInt(NUM_CLICK_REPLICATES, 1));
    conf.setBoolean(IGNORE_FEATURE_VALUE, config.getBoolean(IGNORE_FEATURE_VALUE, false));
    int nblocks = config.getInt(NUM_BLOCKS, 0);
    conf.setInt(NUM_BLOCKS, nblocks);
    _logger.info("Running the preparation job of admm with map.key = " + mapKey + " and num.blocks=" + nblocks);
    AvroUtils.runAvroJob(conf);
  }
  public static class RegressionPrepareMapper extends AvroMapper<GenericData.Record, RegressionPrepareOutput>
  {
    String  _mapkey;
    int     _nblocks;
    int     _numClickReplicates;
    boolean _ignoreValue;

    @Override
    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _mapkey = conf.get(MAP_KEY, "");
      _nblocks = conf.getInt(NUM_BLOCKS, 0);
      _logger.info("nblocks=" + _nblocks);
      _ignoreValue = conf.getBoolean(IGNORE_FEATURE_VALUE, false);
      _numClickReplicates = conf.getInt(NUM_CLICK_REPLICATES, 1);
    }

    @Override
    public void map(GenericData.Record data,
                    AvroCollector<RegressionPrepareOutput> collector,
                    Reporter reporter) throws IOException
    {
      String mapkey = "";
      if (!_mapkey.equals(""))
      {
        if (data.get(_mapkey) == null)
        {
          throw new IOException("map.key is wrongly specified! No such key exists in some lines of the data!");
        }
        mapkey = data.get(_mapkey).toString();
      }
      else
      {
        // if not specified, generate the key by a random number
        mapkey = String.valueOf((int) Math.floor(Math.random() * _nblocks));
      }
      RegressionPrepareOutput outData = new RegressionPrepareOutput();
      outData.key = mapkey;
      // handle response
      int response = Util.getResponseAvro(data);
      outData.response = response;
      List<feature> newfeatures = new ArrayList<feature>();
      // Make sure format in feature is correct
      Object temp = data.get("features");
      if (temp == null)
      {
        throw new IOException("features is null");
      }
      if (!(temp instanceof List))
      {
        throw new IOException("features is not a list");
      }
      List<?> features = (List<?>) temp;
      int m = features.size();
      for (int i = 0; i < m; i++)
      {
        temp = features.get(i);
        if (!(temp instanceof GenericData.Record))
        {
          throw new IOException("features[" + i + "] is not a record");
        }
        GenericData.Record featureRecord = (GenericData.Record) temp;
        String name = Util.getStringAvro(featureRecord, "name", false);
        String term = Util.getStringAvro(featureRecord, "term", true);
        float Value = 1f;
        if (!_ignoreValue)
        {
          Value = (float) Util.getDoubleAvro(featureRecord, "value");
        }
        feature newfeature = new feature();
        newfeature.name = name;
        newfeature.term = term;
        newfeature.value = Value;
        newfeatures.add(newfeature);
      }
      outData.features = newfeatures;
      double weight = 1.0;
      if (data.get("weight") != null)
      {
        weight = Util.getDoubleAvro(data, "weight");
      }
      if (Util.getIntAvro(data, "response") == 1)
      {
        weight = weight / _numClickReplicates;
      }
      outData.weight = (float) weight;

      double offset = 0.0;
      if (data.get("offset") != null)
      {
        offset = Util.getDoubleAvro(data, "offset");
      }
      outData.offset = (float) offset;

      if (_mapkey.equals("") && response == 1)
      {
        // generate click replicates to get better consensus
        int partitionId = Integer.parseInt(mapkey);
        for (int i = 0; i < _numClickReplicates; i++)
        {
          if (partitionId >= _nblocks)
          {
            partitionId = partitionId - _nblocks;
          }
          outData.key = String.valueOf(partitionId);
          collector.collect(outData);
          partitionId++;
        }
      }
      else
      {
        collector.collect(outData);
      }
    }
  }
}
