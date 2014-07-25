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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.avro.Schema;
import org.apache.avro.Schema.Type;
import org.apache.avro.generic.GenericData;
import org.apache.avro.mapred.AvroCollector;
import org.apache.avro.mapred.AvroJob;
import org.apache.avro.mapred.AvroMapper;
import org.apache.avro.mapred.AvroReducer;
import org.apache.avro.mapred.Pair;
import org.apache.avro.util.Utf8;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;
import org.apache.log4j.Logger;

import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.regression.avro.RegressionPrepareOutput;
import com.linkedin.mlease.avro.LinearModelWithVarAvro;
import com.linkedin.mlease.regression.consumers.ReadLambdaMapConsumer;
import com.linkedin.mlease.regression.liblinearfunc.LibLinear;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearBinaryDataset;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearDataset;
import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.AvroConsumer;
import com.linkedin.mapred.AvroDistributedCacheFileReader;
import com.linkedin.mapred.AvroUtils;
import com.linkedin.mapred.JobConfig;

/**
 * Do per-item model training where you can specify variances for any features through a feature map file
 * Also have supports for prior mean/variance for intercept
 * NOTE: intercept lambdas and default lambdas are both arrays of values that will be crossed to determine one config
 * The logic is as following:
 * Intercept Prior Mean: first look up in the prior mean map, if null then use intercept.default.prior.mean
 * Intercept Prior Var: intercept.lambdas
 * Feature Prior Mean: always 0
 * Feature Prior Var: first look up in the lambda map, if null then use values in default.lambdas
 * Output: For each combination of intercept lambda and default lambda, output a trained model for each key
 *
 */
public class ItemModelTrain extends AbstractAvroJob
{
  public static final Logger _logger                  = Logger.getLogger(ItemModelTrain.class);
  public static final String INTERCEPT_PRIOR_MEAN_MAP = "intercept.prior.mean.map";
  public static final String INTERCEPT_DEFAULT_PRIOR_MEAN   = "intercept.default.prior.mean";
  public static final String INTERCEPT_LAMBDAS         = "intercept.lambdas";
  public static final String DEFAULT_LAMBDAS           = "default.lambdas";
  public static final String LAMBDA_MAP               = "lambda.map";
  public static final String OUTPUT_MODEL_PATH        = "output.model.path";
  public static final String REPORT_FREQUENCY         = "report.frequency";
  public static final String LIBLINEAR_INTERCEPT_KEY = "(INTERCEPT)";
  public static final String LIBLINEAR_EPSILON       = "liblinear.epsilon";
  public static final String BINARY_FEATURE          = "binary.feature";
  public static final String SHORT_FEATURE_INDEX     = "short.feature.index";
  public static final String COMPUTE_VAR             = "compute.var";
  public static final String REMOVE_TMP_DIR          = "remove.tmp.dir";
  public ItemModelTrain(String name, JobConfig config)
  {
    super(name, config);
  }

  @Override
  public void run() throws Exception
  {
    JobConfig props = super.getJobConfig();
    _logger.info("Start training per-key naive logistic regression model...");
    String outBasePath = props.getString(OUTPUT_MODEL_PATH);
    String outpath = outBasePath + "/models";
    props.put("output.path", outpath);
    JobConf conf =
        createJobConf(ItemModelTrainMapper.class,
                      ItemModelTrainReducer.class,
                      Pair.getPairSchema(Schema.create(Type.STRING),
                                         RegressionPrepareOutput.SCHEMA$),
                      LinearModelWithVarAvro.SCHEMA$);
    // set up conf
    String interceptPriorMeanMap = props.getString(INTERCEPT_PRIOR_MEAN_MAP,"");
    if (!interceptPriorMeanMap.equals(""))
    {
      AvroUtils.addAvroCacheFilesAndSetTheProperty(conf, new Path(interceptPriorMeanMap), INTERCEPT_PRIOR_MEAN_MAP);
    }
    String lambdaMap = props.getString(LAMBDA_MAP,"");
    if (!lambdaMap.equals(""))
    {
      AvroUtils.addAvroCacheFilesAndSetTheProperty(conf, new Path(lambdaMap), LAMBDA_MAP);
    }
    conf.setFloat(INTERCEPT_DEFAULT_PRIOR_MEAN, (float)props.getDouble(INTERCEPT_DEFAULT_PRIOR_MEAN,0));
    conf.set(INTERCEPT_LAMBDAS,props.get(INTERCEPT_LAMBDAS));
    conf.set(DEFAULT_LAMBDAS,props.get(DEFAULT_LAMBDAS));
    conf.setLong(REPORT_FREQUENCY, props.getLong(REPORT_FREQUENCY, 1000000));
    conf.setFloat(LIBLINEAR_EPSILON, (float) props.getDouble(LIBLINEAR_EPSILON, 0.001f));
    conf.setBoolean(COMPUTE_VAR, props.getBoolean(COMPUTE_VAR,false));
    conf.setBoolean(BINARY_FEATURE, props.getBoolean(BINARY_FEATURE, false));
    conf.setBoolean(SHORT_FEATURE_INDEX, props.getBoolean(SHORT_FEATURE_INDEX, false));
    // run job
    AvroUtils.runAvroJob(conf);
    boolean removeTmpDir = props.getBoolean(REMOVE_TMP_DIR, true);
    if (removeTmpDir)
    {
      FileSystem fs = FileSystem.get(conf);
      fs.delete(new Path(outBasePath + "/tmp-data"), true);
    }
  }

  public static class ItemModelTrainMapper extends AvroMapper<RegressionPrepareOutput, Pair<String, RegressionPrepareOutput>>
  {
    @Override
    public void map(RegressionPrepareOutput data,
                    AvroCollector<Pair<String, RegressionPrepareOutput>> collector,
                    Reporter reporter) throws IOException
    {
      String key = data.key.toString();
      Pair<String, RegressionPrepareOutput> outPair =
          new Pair<String, RegressionPrepareOutput>(key, data);
      collector.collect(outPair);
    }
  }
  
  public static class ItemModelTrainReducer extends AvroReducer<Utf8, RegressionPrepareOutput, GenericData.Record>
  {
    private double  _interceptDefaultPriorMean;
    private List<Float> _interceptLambdas;
    private List<Float> _defaultLambdas;
    
    private long    _reportfreq;
    private boolean _binaryFeature;
    private boolean _shortFeatureIndex;
    private boolean _computeVar;
    private float   _liblinearEpsilon;

    private Map<String, Double> _interceptPriorMeanMap = new HashMap<String, Double>();
    private Map<String, Double> _priorVarMap = new HashMap<String, Double>();
    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _interceptDefaultPriorMean = conf.getFloat(INTERCEPT_DEFAULT_PRIOR_MEAN, 0);
      _interceptLambdas = getValues(conf.getStrings(INTERCEPT_LAMBDAS));
      _defaultLambdas = getValues(conf.getStrings(DEFAULT_LAMBDAS));
      
      _reportfreq = conf.getLong(REPORT_FREQUENCY, 1000000);
      _binaryFeature = conf.getBoolean(BINARY_FEATURE, false);
      _shortFeatureIndex = conf.getBoolean(SHORT_FEATURE_INDEX, false);
      _liblinearEpsilon = conf.getFloat(LIBLINEAR_EPSILON, 0.01f);
      _computeVar = conf.getBoolean(COMPUTE_VAR, false);
      
      // intercept prior mean per campaign
      if (!conf.get(INTERCEPT_PRIOR_MEAN_MAP,"").equals(""))
      {
        try
        {
          ReadPriorMeanMapConsumer priorMeanConsumer = new ReadPriorMeanMapConsumer();
          AvroDistributedCacheFileReader priorMeanReader =
              new AvroDistributedCacheFileReader(new JobConf(conf), true);
          priorMeanReader.build(conf.get(INTERCEPT_PRIOR_MEAN_MAP), priorMeanConsumer);
          _interceptPriorMeanMap = priorMeanConsumer.get();
          _logger.info("loaded intercept prior mean map, size=" + _interceptPriorMeanMap.size());
        } catch (Exception e)
        {
          e.printStackTrace();
          throw new RuntimeException("Can't load intercept prior mean map, error="+e);
        }
      }
      
      // prior variance map for each features
      String lambdaMap = conf.get(LAMBDA_MAP, "");
      if (!lambdaMap.equals(""))
      {
        ReadLambdaMapConsumer lambdaMapConsumer = new ReadLambdaMapConsumer();
        AvroDistributedCacheFileReader lambdaMapReader = new AvroDistributedCacheFileReader(new JobConf(conf), false);
        try
        {
          lambdaMapReader.build(lambdaMap, lambdaMapConsumer);
          lambdaMapConsumer.done();
          for (String k : lambdaMapConsumer.get().keySet())
          {
            double tmp = lambdaMapConsumer.get().get(k);
            tmp = 1.0 / tmp;
            _priorVarMap.put(k, tmp);
          }
          _logger.info("loaded prior variance map, size=" + _priorVarMap.size());
        }
        catch (IOException e)
        {
          e.printStackTrace();
          throw new RuntimeException("Can't load lambda map, error="+e);
        }
      }      
    }

    @Override
    public void reduce(Utf8 key,
                       Iterable<RegressionPrepareOutput> values,
                       AvroCollector<GenericData.Record> collector,
                       Reporter reporter) throws IOException
    {
      // Prepare the data set
      LibLinearDataset dataset;
      if (_binaryFeature)
      {
        dataset = new LibLinearBinaryDataset(1.0, _shortFeatureIndex);
      }
      else
      {
        dataset = new LibLinearDataset(1.0);
      }
      for (RegressionPrepareOutput value : values)
      {
        dataset.addInstanceAvro(value);
      }
      dataset.finish();
      // First determine the prior mean for the intercept
      Map<String, Double> priorMeanMap = new HashMap<String, Double>();
      double interceptPriorMean = _interceptDefaultPriorMean;
      if (_interceptPriorMeanMap.containsKey(key.toString()))
      {
        interceptPriorMean = _interceptPriorMeanMap.get(key.toString());
        reporter.incrCounter("ItemModelTrainV3", "Found intercept prior mean in intercept prior mean map", 1);
      }
      priorMeanMap.put(LibLinearDataset.INTERCEPT_NAME, interceptPriorMean);
      
      // now cross product the lambdas for intercept and default
      for (float interceptLambda : _interceptLambdas)
        for (float defaultLambda : _defaultLambdas)
        {
          _priorVarMap.put(LibLinearDataset.INTERCEPT_NAME, 1.0/interceptLambda);
          GenericData.Record output = new GenericData.Record(LinearModelWithVarAvro.SCHEMA$);
          // Run liblinear
          LibLinear liblinear = new LibLinear();
          liblinear.setReporter(reporter, _reportfreq);
          String option = "epsilon=" + String.valueOf(_liblinearEpsilon);
          try
          {
            liblinear.train(dataset, null, priorMeanMap, _priorVarMap, 0, 1.0 / defaultLambda, option, _computeVar);
            LinearModel model = liblinear.getLinearModel();
            
            output.put("key", String.valueOf(interceptLambda) + ":" + String.valueOf(defaultLambda)+ "#" + key);
            output.put("model", model.toAvro(LIBLINEAR_INTERCEPT_KEY));
            if (_computeVar)
            {
              LinearModel posteriorVar = new LinearModel(LIBLINEAR_INTERCEPT_KEY,liblinear.getPostVarMap());
              output.put("posteriorVar", posteriorVar.toAvro(LIBLINEAR_INTERCEPT_KEY));
            } else
            {
              output.put("posteriorVar", new LinearModel().toAvro(LIBLINEAR_INTERCEPT_KEY));
            }
          }
          catch (Exception e)
          {
            // output everything to debug
            _logger.info("Dataset size=" + dataset.y.length);
            _logger.info("Number of features=" + dataset.nFeatures());
            _logger.info("Model size=" + liblinear.getParamMap().size());
            _logger.info("bias=" + liblinear.bias);
            _logger.info("Model:");
            for (String k : liblinear.getParamMap().keySet())
            {
              _logger.info(k + " " + liblinear.getParamMap().get(k).toString());
            }
            throw new IOException("Model fitting error!", e);
          }
          collector.collect(output);
        }     
    }
    public class ReadPriorMeanMapConsumer implements AvroConsumer<Map<String, Double>>
    {
      private Map<String, Double> _result = new HashMap<String, Double>();
      @Override
      public void consume(Object object)
      {
        Pair record = (Pair) object;
        _result.put(record.key().toString(), Double.parseDouble(record.value().toString()));
      }
      @Override
      public void done()
      {
      }
      @Override
      public Map<String, Double> get() throws IllegalStateException
      {
        return _result;
      }
    }
  }
  private static List<Float> getValues(String[] strings)
  {
    List<Float> result = new ArrayList<Float>();
    for (String s : strings)
    {
      result.add(Float.parseFloat(s));
    }
    return result;
  }
}

