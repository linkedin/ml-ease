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

package com.linkedin.mlease.utils;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.avro.file.DataFileWriter;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.mapred.JobConf;
import org.apache.log4j.Logger;

import com.linkedin.mlease.avro.LinearModelAvro;
import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.regression.consumers.MeanLinearModelConsumer;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearDataset;
import com.linkedin.mapred.AvroHdfsFileReader;
import com.linkedin.mapred.AvroHdfsFileWriter;


public class LinearModelUtils
{
  private static final Logger _logger = Logger.getLogger(LinearModelUtils.class);
  public static void writeLinearModel(JobConf conf, String path, Map<String, LinearModel> models) throws IOException
  {
    AvroHdfsFileWriter<GenericRecord> writer =
        new AvroHdfsFileWriter<GenericRecord>(conf, path, LinearModelAvro.SCHEMA$);
    DataFileWriter<GenericRecord> recordWriter = writer.get();
    for (String k : models.keySet())
    {
      GenericRecord record = new GenericData.Record(LinearModelAvro.SCHEMA$);
      List modellist = models.get(k).toAvro(LibLinearDataset.INTERCEPT_NAME);
      record.put("key", k);
      record.put("model", modellist);
      recordWriter.append(record);
    }
    recordWriter.close();
  }

  public static void writeLinearModel(JobConf conf, String path, LinearModel model, String modelkey) throws IOException
  {
    AvroHdfsFileWriter<GenericRecord> writer =
        new AvroHdfsFileWriter<GenericRecord>(conf, path, LinearModelAvro.SCHEMA$);
    DataFileWriter<GenericRecord> recordWriter = writer.get();
    List modellist = model.toAvro(LibLinearDataset.INTERCEPT_NAME);
    GenericData.Record valuemap = new GenericData.Record(LinearModelAvro.SCHEMA$);
    valuemap.put("key", modelkey);
    valuemap.put("model", modellist);
    recordWriter.append(valuemap);
    recordWriter.close();
  }
  
  public static Map<String, LinearModel> meanModel(JobConf conf, String modelPath,
                                                   int nblocks,
                                                   int lambdasize,
                                                   boolean check) throws Exception
  {
    AvroHdfsFileReader reader = new AvroHdfsFileReader(conf);
    MeanLinearModelConsumer consumer = new MeanLinearModelConsumer(nblocks);
    reader.build(modelPath, consumer);
    consumer.done();
    if (check)
    {
      _logger.info("Number of successful models=" + String.valueOf(consumer.getCounter()));
      if (consumer.getCounter() != lambdasize * nblocks)
      {
        throw new RuntimeException("Some models failed!");
      }
    }
    return consumer.get();
  }
}
