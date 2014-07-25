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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.Reporter;
import org.apache.log4j.Logger;

import com.linkedin.mlease.avro.LinearModelAvro;
import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.regression.avro.RegressionPrepareOutput;
import com.linkedin.mlease.regression.consumers.ReadLambdaMapConsumer;
import com.linkedin.mlease.regression.consumers.ReadPartitionIdAssignmentConsumer;
import com.linkedin.mlease.regression.jobs.PartitionIdAssigner.PartitionIdAssignerCombiner;
import com.linkedin.mlease.regression.jobs.PartitionIdAssigner.PartitionIdAssignerMapper;
import com.linkedin.mlease.regression.jobs.PartitionIdAssigner.PartitionIdAssignerReducer;
import com.linkedin.mlease.regression.liblinearfunc.LibLinear;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearBinaryDataset;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearDataset;
import com.linkedin.mlease.utils.LinearModelUtils;
import com.linkedin.mlease.utils.Util;
import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.AvroDistributedCacheFileReader;
import com.linkedin.mapred.AvroHdfsFileReader;
import com.linkedin.mapred.AvroUtils;
import com.linkedin.mapred.JobConfig;
/**
 * Fit the logistic regression model with penalty l2 using simple divide, fitting and
 * taking mean approach This is a very naive large-scale model fitting approach. For
 * better approach please use AdmmTrain.java
 * 
 * 
 */
public class RegressionNaiveTrain extends AbstractAvroJob
{
  private static final Logger _logger                 = Logger.getLogger(RegressionNaiveTrain.class);
  public static final String NUM_BLOCKS              = "num.blocks";
  public static final String LAMBDA                  = "lambda";
  public static final String PRIOR_MEAN              = "prior.mean";
  public static final String PENALIZE_INTERCEPT      = "penalize.intercept";
  public static final String HAS_INTERCEPT           = "has.intercept";
  public static final String INTERCEPT_KEY           = "intercept.key";
  public static final String OUTPUT_BASE_PATH        = "output.base.path";
  public static final String REPORT_FREQUENCY        = "report.frequency";
  public static final String REMOVE_TMP_DIR          = "remove.tmp.dir";
  public static final String LIBLINEAR_INTERCEPT_KEY = "(INTERCEPT)";
  public static final String LIBLINEAR_EPSILON       = "liblinear.epsilon";
  public static final String LAMBDA_MAP              = "lambda.map";
  public static final String BINARY_FEATURE          = "binary.feature";
  public static final String SHORT_FEATURE_INDEX     = "short.feature.index";
  public static final String HEAVY_PER_ITEM_TRAIN    = "heavy.per.item.train";
  // true if random split; false if say per-creative model
  public static final String COMPUTE_MODEL_MEAN      = "compute.model.mean";              
  // threshold: if number of events < threshold, then don't train
  public static final String DATA_SIZE_THRESHOLD = "data.size.threshold";
  // not needed in config
  public static final String PARTITION_ID_PATH       = "partition.id.path";
  
  public RegressionNaiveTrain(String jobId, JobConfig config)
  {
    super(jobId, config);
  }
  @Override
  public void run() throws Exception
  {
    JobConfig props = super.getJobConfig();
    String outBasePath = props.getString(OUTPUT_BASE_PATH);
    boolean heavyPerItemTrain = props.getBoolean(HEAVY_PER_ITEM_TRAIN,false);

    String partitionIdPath = "";
    if (heavyPerItemTrain)
    {
      partitionIdPath = outBasePath + "/partitionIds";
      props.put(AbstractAvroJob.OUTPUT_PATH, partitionIdPath);
      JobConf conf =
          createJobConf(PartitionIdAssignerMapper.class,
                        PartitionIdAssignerReducer.class,
                        PartitionIdAssignerCombiner.class,
                        Pair.getPairSchema(Schema.create(Type.STRING),
                                           Schema.create(Type.INT)),
                        Pair.getPairSchema(Schema.create(Type.STRING),
                                           Schema.create(Type.INT))
                        );
      conf.set(LAMBDA, props.getString(LAMBDA));
      AvroJob.setInputSchema(conf, RegressionPrepareOutput.SCHEMA$);
      conf.setNumReduceTasks(1);
      AvroUtils.runAvroJob(conf);
    }
    _logger.info("Start training per-key naive logistic regression model...");    
    String outpath = outBasePath + "/models";
    props.put(AbstractAvroJob.OUTPUT_PATH, outpath);
    JobConf conf =
        createJobConf(NaiveMapper.class,
                      NaiveReducer.class,
                      Pair.getPairSchema(Schema.create(Type.STRING),
                                         RegressionPrepareOutput.SCHEMA$),
                      LinearModelAvro.SCHEMA$);
    // set up conf
    boolean computeModelMean = props.getBoolean(COMPUTE_MODEL_MEAN, true);
    int nblocks = -1;
    if (computeModelMean)
    {
      nblocks = props.getInt(NUM_BLOCKS);
      conf.setInt(NUM_BLOCKS, nblocks);
    }
    List<String> lambdastr = props.getStringList(LAMBDA, ",");
    conf.set(LAMBDA, props.getString(LAMBDA));
    conf.setFloat(PRIOR_MEAN, props.getFloat(PRIOR_MEAN,0.0));
    conf.setBoolean(PENALIZE_INTERCEPT, props.getBoolean(PENALIZE_INTERCEPT, false));
    conf.setBoolean(HAS_INTERCEPT, props.getBoolean(HAS_INTERCEPT, true));
    conf.set(INTERCEPT_KEY, props.getString(INTERCEPT_KEY, LIBLINEAR_INTERCEPT_KEY));
    conf.setLong(REPORT_FREQUENCY, props.getLong(REPORT_FREQUENCY, 1000000));
    boolean removeTmpDir = props.getBoolean(REMOVE_TMP_DIR, true);
    conf.setFloat(LIBLINEAR_EPSILON, props.getFloat(LIBLINEAR_EPSILON, 0.001f));
    String lambdaMap = props.getString(LAMBDA_MAP, "");
    conf.set(LAMBDA_MAP, lambdaMap);
    if (!lambdaMap.equals(""))
    {
      AvroUtils.addAvroCacheFiles(conf, new Path(lambdaMap));
    }
    conf.setBoolean(BINARY_FEATURE, props.getBoolean(BINARY_FEATURE, false));
    conf.setBoolean(SHORT_FEATURE_INDEX, props.getBoolean(SHORT_FEATURE_INDEX, false));
    // set up lambda
    Set<Float> lambdaSet = new HashSet<Float>();
    for (String l : lambdastr)
    {
      lambdaSet.add(Float.parseFloat(l));
    }

    conf.setInt(DATA_SIZE_THRESHOLD, props.getInt(DATA_SIZE_THRESHOLD,0));
    // set up partition id
    if (heavyPerItemTrain && !partitionIdPath.equals(""))
    {
      conf.set(PARTITION_ID_PATH, partitionIdPath);
      AvroHdfsFileReader reader = new AvroHdfsFileReader(conf);
      ReadPartitionIdAssignmentConsumer consumer = new ReadPartitionIdAssignmentConsumer();
      reader.build(partitionIdPath, consumer);
      Map<String, Integer> partitionIdMap = consumer.get();
      int maxPartitionId = 0;
      for (int v : partitionIdMap.values())
      {
        if (v>maxPartitionId)
        {
          maxPartitionId = v;
        }
      }
      AvroUtils.addAvroCacheFiles(conf, new Path(partitionIdPath));
      conf.setNumReduceTasks(maxPartitionId+1);
      conf.setPartitionerClass(NaivePartitioner.class);
    }
    // run job
    AvroJob.setInputSchema(conf, RegressionPrepareOutput.SCHEMA$);
    AvroUtils.runAvroJob(conf);
    // Compute Mean
    if (computeModelMean)
    {
      Map<String, LinearModel> betabar =
          LinearModelUtils.meanModel(conf, outpath, nblocks, lambdaSet.size(), true);
      // Output the mean for each lambda
      // write z into file
      String finalOutPath = outBasePath + "/final-model/part-r-00000.avro";
      LinearModelUtils.writeLinearModel(conf, finalOutPath, betabar);
    }
    // remove tmp dir
    if (removeTmpDir)
    {
      FileSystem fs = FileSystem.get(conf);
      fs.delete(new Path(outBasePath + "/tmp-data"), true);
    }
  }

  public static class NaiveMapper extends
      AvroMapper<RegressionPrepareOutput, Pair<String, RegressionPrepareOutput>>
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
                    AvroCollector<Pair<String, RegressionPrepareOutput>> collector,
                    Reporter reporter) throws IOException
    {
      String key = data.key.toString();
      for (float lambda : _lambdaSet)
      {
        String newkey = String.valueOf(lambda) + "#" + key;
        data.key = newkey;
        Pair<String, RegressionPrepareOutput> outPair =
            new Pair<String, RegressionPrepareOutput>(newkey, data);
        collector.collect(outPair);
      }
    }
  }

  public static class NaivePartitioner implements
  Partitioner<AvroKey<String>, AvroValue<Integer>>
  {
    Map<String, Integer> _partitionIdMap = null;
    @Override
    public void configure(JobConf conf)
    {
      AvroDistributedCacheFileReader reader = new AvroDistributedCacheFileReader(conf);
      ReadPartitionIdAssignmentConsumer consumer = new ReadPartitionIdAssignmentConsumer();
      String partitionIdPath = conf.get(PARTITION_ID_PATH,"");
      if (!partitionIdPath.equals(""))
      {
        try
        {
          reader.build(partitionIdPath, consumer);
          _partitionIdMap = consumer.get();
        }
        catch (IOException e)
        {
          e.printStackTrace();
        }
      }
    }

    @Override
    public int getPartition(AvroKey<String> key,
                            AvroValue<Integer> value,
                            int numPartitions)
    {
      String k = key.datum().toString();
      if (_partitionIdMap!=null)
      {
        if (_partitionIdMap.containsKey(k))
        {
          int partitionId = _partitionIdMap.get(k);
          return partitionId % numPartitions;
        }
      }
      return Math.abs(k.hashCode()) % numPartitions;
    }
  }
  
  public static class NaiveReducer extends
      AvroReducer<Utf8, RegressionPrepareOutput, GenericData.Record>
  {
    String                        _interceptKey;
    long                          _reportfreq;
    boolean                       _hasIntercept;
    boolean                       _penalizeIntercept;
    boolean                       _binaryFeature;
    boolean                       _shortFeatureIndex;
    float                         _liblinearEpsilon;
    int                           _dataSizeThreshold;
    Map<String, Double>           _lambdaMap          = null;
    JobConf                       _conf;
    private ReadLambdaMapConsumer _lambdaMapConsumer = new ReadLambdaMapConsumer();
    private double               _priorMean;

    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _interceptKey = conf.get(INTERCEPT_KEY, LIBLINEAR_INTERCEPT_KEY);
      _reportfreq = conf.getLong(REPORT_FREQUENCY, 1000000);
      _hasIntercept = conf.getBoolean(HAS_INTERCEPT, true);
      _binaryFeature = conf.getBoolean(BINARY_FEATURE, false);
      _shortFeatureIndex = conf.getBoolean(SHORT_FEATURE_INDEX, false);
      _liblinearEpsilon = conf.getFloat(LIBLINEAR_EPSILON, 0.01f);
      _penalizeIntercept = conf.getBoolean(PENALIZE_INTERCEPT, false);
      _dataSizeThreshold = conf.getInt(DATA_SIZE_THRESHOLD, 0);
      _priorMean = conf.getFloat(PRIOR_MEAN, 0.0f);
      String lambda_map = conf.get(LAMBDA_MAP, "");
      if (!lambda_map.equals(""))
      {
        AvroDistributedCacheFileReader lambdaMapReader =
            new AvroDistributedCacheFileReader(new JobConf(conf));
        try
        {
          lambdaMapReader.build(lambda_map, _lambdaMapConsumer);
        }
        catch (IOException e)
        {
          e.printStackTrace();
        }
      }
      _lambdaMapConsumer.done();
      _lambdaMap = new HashMap<String, Double>();
      for (String k : _lambdaMapConsumer.get().keySet())
      {
        double tmp = _lambdaMapConsumer.get().get(k);
        tmp = 1.0 / tmp;
        _lambdaMap.put(k, tmp);
      }
      if (!_penalizeIntercept)
      {
        _lambdaMap.put(_interceptKey, 100000.0); // very large variance for intercept
      }
      _conf = new JobConf(conf);
    }

    @Override
    public void reduce(Utf8 key,
                       Iterable<RegressionPrepareOutput> values,
                       AvroCollector<GenericData.Record> collector,
                       Reporter reporter) throws IOException
    {
      _logger.info("Memory usage before loading the data:");
      _logger.info("free memory="+Runtime.getRuntime().freeMemory());
      _logger.info("max memory="+Runtime.getRuntime().maxMemory());
      _logger.info("total memory="+Runtime.getRuntime().totalMemory());
      float lambda = Float.parseFloat(Util.getLambda(key.toString()));
      // Prepare the data set
      LibLinearDataset dataset;
      double bias = 0;
      if (_hasIntercept)
      {
        bias = 1;
      }
      if (_binaryFeature)
      {
        dataset = new LibLinearBinaryDataset(bias, _shortFeatureIndex);
      }
      else
      {
        dataset = new LibLinearDataset(bias);
      }
      for (RegressionPrepareOutput value : values)
      {
        dataset.addInstanceAvro(value);
      }
      System.gc();
      dataset.finish();
      if (dataset.y.length<_dataSizeThreshold)
      {
        return;
      }
      _logger.info("Memory usage after loading the data:");
      _logger.info("free memory="+Runtime.getRuntime().freeMemory());
      _logger.info("max memory="+Runtime.getRuntime().maxMemory());
      _logger.info("total memory="+Runtime.getRuntime().totalMemory());
      
      GenericData.Record output = new GenericData.Record(LinearModelAvro.SCHEMA$);
      // Run liblinear
      LibLinear liblinear = new LibLinear();
      liblinear.setReporter(reporter, _reportfreq);
      String option = "epsilon=" + String.valueOf(_liblinearEpsilon);
      try
      {
        liblinear.train(dataset, null, null, _lambdaMap, _priorMean, 1.0 / lambda, option);
        LinearModel model = liblinear.getLinearModel();
        output.put("key", key);
        output.put("model", model.toAvro(LIBLINEAR_INTERCEPT_KEY));
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

}

