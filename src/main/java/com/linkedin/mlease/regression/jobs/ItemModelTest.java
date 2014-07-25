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
import java.net.URISyntaxException;
import java.util.LinkedList;
import java.util.List;

import org.apache.avro.Schema;
import org.apache.avro.Schema.Type;
import org.apache.avro.generic.GenericData;
import org.apache.avro.mapred.AvroCollector;
import org.apache.avro.mapred.AvroJob;
import org.apache.avro.mapred.AvroKey;
import org.apache.avro.mapred.AvroMapper;
import org.apache.avro.mapred.AvroReducer;
import org.apache.avro.mapred.AvroValue;
import org.apache.avro.mapred.Pair;
import org.apache.avro.util.Utf8;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.Reporter;
import org.apache.log4j.Logger;

import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.regression.consumers.ReadLinearModelConsumer;
import com.linkedin.mlease.utils.Util;
import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.AvroDistributedCacheFileReader;
import com.linkedin.mapred.AvroUtils;
import com.linkedin.mapred.JobConfig;


public class ItemModelTest extends AbstractAvroJob
{
  public static final Logger _logger          = Logger.getLogger(ItemModelTest.class);
  public static final String MODEL_PATH       = "model.path";
  public static final String LAMBDA           = "lambda";
  public static final String ITEM_KEY         = "item.key";
  public static final String OUTPUT_BASE_PATH = "output.base.path";
  public static final String BINARY_FEATURE   = "binary.feature";
  public static final String NUM_REDUCERS     = "num.reducers";

  public ItemModelTest(String name, JobConfig config)
  {
    super(name, config);
  }

  @Override
  public void run() throws Exception
  {
    JobConfig props = super.getJobConfig();
    List<String> lambdastr = props.getStringList(LAMBDA, ",");
    String outBasePath = props.getString(OUTPUT_BASE_PATH);
    for (String lambda : lambdastr)
    {
      String outPath = outBasePath + "/lambda-" + lambda;
      props.put("output.path", outPath);
      JobConf conf = createJobConf(PerItemTestMapper.class, PerItemTestReducer.class);
      AvroUtils.addAvroCacheFilesAndSetTheProperty(conf,
                                                   new Path(props.get(MODEL_PATH)),
                                                   MODEL_PATH);
      conf.set(ITEM_KEY, props.getString(ITEM_KEY));
      conf.setFloat(LAMBDA, Float.parseFloat(lambda));
      conf.setBoolean(BINARY_FEATURE, props.getBoolean(BINARY_FEATURE, false));
      conf.setPartitionerClass(PerItemTestPartitioner.class);
      conf.setInt(NUM_REDUCERS, conf.getNumReduceTasks());
      AvroUtils.runAvroJob(conf);
    }
  }

  public static class PerItemTestMapper extends
      AvroMapper<GenericData.Record, Pair<String, GenericData.Record>>
  {
    private String _itemKey;

    @Override
    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _itemKey = conf.get(ITEM_KEY, "");
    }

    @Override
    public void map(GenericData.Record data,
                    AvroCollector<Pair<String, GenericData.Record>> collector,
                    Reporter reporter) throws IOException
    {
      if (data.get(_itemKey) == null)
      {
        throw new IOException("data does not contain the column" + _itemKey);
      }
      String itemKey = data.get(_itemKey).toString();
      collector.collect(new Pair<String, GenericData.Record>(itemKey, data));
    }

  }

  public static class PerItemTestPartitioner implements
      Partitioner<AvroKey<String>, AvroValue<GenericData.Record>>
  {
    @Override
    public void configure(JobConf conf)
    {
    }

    @Override
    public int getPartition(AvroKey<String> key,
                            AvroValue<GenericData.Record> value,
                            int numPartitions)
    {
      return Math.abs(key.datum().hashCode()) % numPartitions;
    }
  }

  public static class PerItemTestReducer extends
      AvroReducer<Utf8, GenericData.Record, GenericData.Record>
  {
    private float                   _lambda      = 0;
    private boolean                 _ignoreValue = false;
    private Schema                  _outputSchema;
    private ReadLinearModelConsumer _consumer;

    @Override
    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _outputSchema = AvroJob.getOutputSchema(conf);
      _lambda = conf.getFloat(LAMBDA, 0);
      _ignoreValue = conf.getBoolean(BINARY_FEATURE, false);
      String modelPath = conf.get(MODEL_PATH, "");
      _logger.info("Going to read model files from distributed cache at:" + modelPath);
      int reduceTaskId = conf.getInt("mapred.task.partition", -1);
      _logger.info("The reduce task id=" + reduceTaskId);
      if (reduceTaskId < 0)
      {
        throw new RuntimeException("Can't read reduce task id from mapred.task.partition!");
      }
      int nReducers = conf.getInt(NUM_REDUCERS, -1);
      String lambdaKey = String.valueOf(_lambda) + "#";
      _consumer = new ReadLinearModelConsumer(lambdaKey, reduceTaskId, nReducers);
      AvroDistributedCacheFileReader modelReader =
          new AvroDistributedCacheFileReader(new JobConf(conf));
      try
      {
        modelReader.build(modelPath, _consumer);
        _consumer.done();
      }
      catch (IOException e)
      {
        throw new RuntimeException("Can't load model, error=" + e);
      }
      _logger.info("Loaded linear models, number of models loaded="
          + _consumer.get().size());
    }

    @Override
    public void reduce(Utf8 key,
                       Iterable<GenericData.Record> values,
                       AvroCollector<GenericData.Record> collector,
                       Reporter reporter) throws IOException
    {
      String modelKey = String.valueOf(_lambda) + "#" + key.toString();
      LinearModel model;
      if (_consumer.get().containsKey(modelKey))
      {
        model = _consumer.get().get(modelKey);
      }
      else
      {
        _logger.info("The model key can not be found in the model. Key=" + modelKey);
        model = new LinearModel();
      }
      for (GenericData.Record data : values)
      {
        float pred =
            (float) model.evalInstanceAvro(data, false, _ignoreValue);
        GenericData.Record output = new GenericData.Record(_outputSchema);
        List<Schema.Field> inputFields = data.getSchema().getFields();
        for (Schema.Field field : inputFields)
        {
          output.put(field.name(), data.get(field.name()));
        }
        output.put("pred", pred);
        collector.collect(output);
      }
    }
  }

  private JobConf createJobConf(Class<? extends AvroMapper> mapperClass,
                                Class<? extends AvroReducer> reducerClass) throws IOException, URISyntaxException
  {
    JobConf conf = createJobConf();
    Schema inputSchema = Util.removeUnion(AvroUtils.getAvroInputSchema(conf));
    if (inputSchema == null)
    {
      throw new IllegalStateException("Input does not have schema info and/or input is missing.");
    }
    _logger.info("Input Schema=" + inputSchema.toString());
    List<Schema.Field> inputFields = inputSchema.getFields();
    Schema.Field predField =
        new Schema.Field("pred", Schema.create(Type.FLOAT), "", null);
    List<Schema.Field> outputFields = new LinkedList<Schema.Field>();
    for (Schema.Field field : inputFields)
    {
      outputFields.add(new Schema.Field(field.name(),
                                        field.schema(),
                                        field.doc(),
                                        null));
    }
    outputFields.add(predField);
    Schema outputSchema =
        Schema.createRecord("PerItemTestOutput",
                            "Test output for PerItemTest",
                            "com.linkedin.lab.regression.avro",
                            false);
    outputSchema.setFields(outputFields);
    AvroJob.setOutputSchema(conf, outputSchema);
    AvroJob.setMapOutputSchema(conf,
                               Pair.getPairSchema(Schema.create(Type.STRING), inputSchema));
    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, reducerClass);
    return conf;
  }
}

