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
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.apache.avro.Schema;
import org.apache.avro.Schema.Type;
import org.apache.avro.generic.GenericData;
import org.apache.avro.mapred.AvroCollector;
import org.apache.avro.mapred.AvroJob;
import org.apache.avro.mapred.AvroMapper;
import org.apache.avro.mapred.AvroOutputFormat;
import org.apache.avro.mapred.AvroReducer;
import org.apache.avro.mapred.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;
import org.apache.log4j.Logger;

import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.regression.consumers.ReadLinearModelConsumer;
import com.linkedin.mlease.utils.Util;
import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.AvroDistributedCacheFileReader;
import com.linkedin.mapred.AvroUtils;
import com.linkedin.mapred.JobConfig;


public class RegressionTest extends AbstractAvroJob
{
  public static final Logger _logger          = Logger.getLogger(RegressionTest.class);
  public static final String MODEL_BASE_PATH  = "model.base.path";
  public static final String LAMBDA           = "lambda";
  public static final String OUTPUT_BASE_PATH = "output.base.path";
  public static final String BINARY_FEATURE   = "binary.feature";
  
  public static final String MODEL_PATH       = "model.path";

  public RegressionTest(String name, JobConfig config)
  {
    super(name, config);
  }

  @Override
  public void run() throws Exception
  {
    JobConfig props = super.getJobConfig();
    JobConf conf = super.createJobConf();
    if (!props.getString("input.paths").equals(""))
    {
      // set up configuration
      _logger.info("Now starting test...");
      List<String> lambdastr = props.getStringList(LAMBDA, ",");
      String outBasePath = props.getString(OUTPUT_BASE_PATH);
      for (String lambda : lambdastr)
      {
        String outPath = outBasePath + "/lambda-" + lambda;
        props.put(AbstractAvroJob.OUTPUT_PATH, outPath);
        conf = createJobConf(AdmmTestMapper.class, AdmmTestReducer.class);
        AvroOutputFormat.setOutputPath(conf, new Path(outPath));
        String modelPath = props.getString(MODEL_BASE_PATH);
        modelPath = modelPath + "/final-model";
        AvroUtils.addAvroCacheFiles(conf, new Path(modelPath));
        conf.set(MODEL_PATH, modelPath);
        conf.setFloat(LAMBDA, Float.parseFloat(lambda));
        conf.setBoolean(BINARY_FEATURE, props.getBoolean(BINARY_FEATURE, false));
        AvroJob.setInputSchema(conf, AvroUtils.getAvroInputSchema(conf));
        AvroUtils.runAvroJob(conf);
      }
      // also do full prediction on best-model if it exists
      FileSystem fs = FileSystem.get(conf);
      String modelPath = props.getString(MODEL_BASE_PATH) + "/best-model";
      if (fs.exists(new Path(modelPath)))
      {
        String outPath = outBasePath + "/best-model";
        props.put(AbstractAvroJob.OUTPUT_PATH, outPath);
        conf = createJobConf(AdmmTestMapper.class, AdmmTestReducer.class);
        AvroOutputFormat.setOutputPath(conf, new Path(outPath));
        AvroUtils.addAvroCacheFiles(conf, new Path(modelPath));
        conf.set(MODEL_PATH, modelPath);
        conf.setFloat(LAMBDA, -1);
        conf.setBoolean(BINARY_FEATURE, props.getBoolean(BINARY_FEATURE, false));
        AvroJob.setInputSchema(conf, AvroUtils.getAvroInputSchema(conf));
        AvroUtils.runAvroJob(conf);
      }
    }
    else
    {
      _logger.info("test.input.paths is empty! So no test will be done!");
    }
  }

  public static class AdmmTestMapper extends
      AvroMapper<GenericData.Record, Pair<Float, GenericData.Record>>
  {
    private ReadLinearModelConsumer _modelConsumer = new ReadLinearModelConsumer();
    private float                   _lambda         = 0;
    private boolean                 _ignoreValue   = false;
    Schema                          _outputSchema;

    @Override
    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _outputSchema = AvroJob.getOutputSchema(conf);
      AvroDistributedCacheFileReader modelReader =
          new AvroDistributedCacheFileReader(new JobConf(conf));
      try
      {
        modelReader.build(conf.get(MODEL_PATH), _modelConsumer);
        _modelConsumer.done();
      }
      catch (IOException e)
      {
        e.printStackTrace();
      }
      _lambda = conf.getFloat(LAMBDA, 0);
      _ignoreValue = conf.getBoolean(BINARY_FEATURE, false);
      _logger.info("Loaded the model for test, size:" + _modelConsumer.get().size());
    }

    @Override
    public void map(GenericData.Record data,
                    AvroCollector<Pair<Float, GenericData.Record>> collector,
                    Reporter reporter) throws IOException
    {
      LinearModel model;
      if (_lambda >= 0)
      {
        model = _modelConsumer.get().get(String.valueOf(_lambda));
      }
      else
      {
        // lambda should be -1 and it should include only 1 model which is the best-model
        // found in train
        Iterator<LinearModel> iter = _modelConsumer.get().values().iterator();
        model = iter.next();
      }
      float pred = (float) model.evalInstanceAvro(data, false, _ignoreValue);
      GenericData.Record output = new GenericData.Record(_outputSchema);
      List<Schema.Field> inputFields = data.getSchema().getFields();
      for (Schema.Field field : inputFields)
      {
        output.put(field.name(), data.get(field.name()));
        _logger.info(field.name() + ": " + data.get(field.name()));
      }
      output.put("pred", pred);
      Pair<Float, GenericData.Record> outPair =
          new Pair<Float, GenericData.Record>(pred, output);
      collector.collect(outPair);
    }
  }

  public static class AdmmTestReducer extends
      AvroReducer<Float, GenericData.Record, GenericData.Record>
  {
    @Override
    public void reduce(Float key,
                       Iterable<GenericData.Record> values,
                       AvroCollector<GenericData.Record> collector,
                       Reporter reporter) throws IOException
    {
      for (GenericData.Record data : values)
      {
        try
        {
          collector.collect(data);
        }
        catch (IOException e)
        {
          throw new IllegalStateException(e);
        }
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
        Schema.createRecord("AdmmTestOutput",
                            "Test output for AdmmTest",
                            "com.linkedin.lab.regression.avro",
                            false);
    outputSchema.setFields(outputFields);
    AvroJob.setOutputSchema(conf, outputSchema);
    AvroJob.setMapOutputSchema(conf,
                               Pair.getPairSchema(Schema.create(Type.FLOAT), outputSchema));
    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, reducerClass);
    return conf;
  }
}
