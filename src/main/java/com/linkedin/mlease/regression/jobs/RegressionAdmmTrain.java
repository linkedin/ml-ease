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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.avro.Schema;
import org.apache.avro.Schema.Type;
import org.apache.avro.file.DataFileStream;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.mapred.AvroCollector;
import org.apache.avro.mapred.AvroJob;
import org.apache.avro.mapred.AvroKey;
import org.apache.avro.mapred.AvroMapper;
import org.apache.avro.mapred.AvroOutputFormat;
import org.apache.avro.mapred.AvroReducer;
import org.apache.avro.mapred.AvroValue;
import org.apache.avro.mapred.Pair;
import org.apache.commons.lang.mutable.MutableFloat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.Reporter;
import org.apache.log4j.Logger;


import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.avro.LinearModelAvro;
import com.linkedin.mlease.regression.avro.LambdaRhoMap;
import com.linkedin.mlease.regression.avro.RegressionPrepareOutput;
import com.linkedin.mlease.regression.avro.RegressionTrainOutput;
import com.linkedin.mlease.regression.avro.SampleTestLoglik;
import com.linkedin.mlease.regression.consumers.FindLinearModelConsumer;
import com.linkedin.mlease.regression.consumers.ReadLambdaMapConsumer;
import com.linkedin.mlease.regression.consumers.ReadLambdaRhoConsumer;
import com.linkedin.mlease.regression.liblinearfunc.LibLinear;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearBinaryDataset;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearDataset;
import com.linkedin.mlease.utils.LinearModelUtils;
import com.linkedin.mlease.utils.Util;
import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.AvroDistributedCacheFileReader;
import com.linkedin.mapred.AvroHdfsFileReader;
import com.linkedin.mapred.AvroHdfsFileWriter;
import com.linkedin.mapred.AvroUtils;
import com.linkedin.mapred.JobConfig;

public class RegressionAdmmTrain extends AbstractAvroJob
{
  public static final Logger _logger                 = Logger.getLogger(RegressionAdmmTrain.class);
  /**
   * Must have configs
   */
  // the output root dir
  public static final String OUTPUT_BASE_PATH        = "output.base.path";
  // test path for drawing test-loglik trajectory over iterations
  public static final String TEST_PATH               = "test.path";
  public static final String NUM_BLOCKS              = "num.blocks";
  public static final String LAMBDA                  = "lambda";
  public static final String NUM_ITERS               = "num.iters";
  //Regularizer L2 vs L1
  public static final String REGULARIZER         = "regularizer";
  /**
   * Optional configs
   */
  public static final String PENALIZE_INTERCEPT      = "penalize.intercept";
  public static final String REMOVE_TMP_DIR          = "remove.tmp.dir";
  public static final String EPSILON                 = "epsilon";
  public static final String LIBLINEAR_EPSILON       = "liblinear.epsilon";
  public static final String LAMBDA_MAP              = "lambda.map";
  public static final String BINARY_FEATURE          = "binary.feature";
  public static final String SHORT_FEATURE_INDEX     = "short.feature.index";
  //Flag for aggressively decreasing liblinear tolerance threshold
  public static final String AGGRESSIVE_LIBLINEAR_EPSILON_DECAY  = "aggressive.liblinear.epsilon.decay";
  // whether to do test-loglik for every iteration?
  public static final String TEST_LOGLIK_PER_ITER    = "test.loglik.per.iter";      
  // default 1, when >1 it means it replicates the clicks by several times to make sure we have better consensus
  public static final String NUM_CLICK_REPLICATES    = "num.click.replicates";
  //initialize.boost.rate: default is 0; if 0  there is no initialization; 
  //if >0, it will do mean model initialization and boost rho to rho*initilize.boost.rate  
  public static final String INITIALIZE_BOOST_RATE      = "initialize.boost.rate";
  //rho.adapt.coefficient: default is 0; if 0 not to adapt rho for each iteration; if > 0, 
  //it will   use rho.adapt.coefficient as decay parameter for exponential 
  //decay of rho at each iteration (except the first iteratio with initialization, rho is boosted with initilize.boost.rate.
  //suggested value is 0.3;
  public static final String RHO_ADAPT_COEFFICIENT = "rho.adapt.coefficient";
  //use it to pass actual rho adaptive rate to reducer at each iteration
  public static final String RHO_ADAPT_RATE = "rho.adapt.rate";
  /**
   * Not for config, but for defining constant strings
   */
  public static final String RHO                     = "rho";  
  public static final String INTERCEPT_KEY           = "intercept.key";
  public static final String U_PATH                  = "u.path";
  public static final String INIT_VALUE_PATH         = "init.value.path";
  public static final String REPORT_FREQUENCY        = "report.frequency";
  public static final String LAMBDA_RHO_MAP          = "lambda.rho.map";   
  // max number of test events
  public static final long   MAX_NTEST_EVENTS        = 1000000;                          
  

  public RegressionAdmmTrain(String jobId, JobConfig config)
  {
    super(jobId, config);
  }
  @Override
  public void run() throws Exception
  {
    _logger.info("Now running Regression Train using ADMM...");
    JobConfig props = super.getJobConfig();
    String outBasePath = props.getString(OUTPUT_BASE_PATH);
    JobConf conf = super.createJobConf();
    
    // Various configs
    int nblocks = props.getInt(NUM_BLOCKS);
    int niter = props.getInt(NUM_ITERS, 10);
    //Aggressive decay of liblinear_epsilon
    boolean aggressiveLiblinearEpsilonDecay = props.getBoolean(AGGRESSIVE_LIBLINEAR_EPSILON_DECAY,false); 
    // Getting the value of the regularizer L1/L2
    int reg = props.getInt(REGULARIZER);
    if((reg!=1) && (reg!=2))
    {
      throw new IOException("Only L1 and L2 regularization supported!");
    }
    int numClickReplicates = props.getInt(NUM_CLICK_REPLICATES, 1);
    boolean ignoreValue = props.getBoolean(BINARY_FEATURE, false);  
    float initializeBoostRate = props.getFloat(INITIALIZE_BOOST_RATE, 0);
    float rhoAdaptCoefficient = props.getFloat(RHO_ADAPT_COEFFICIENT, 0);
    
    // handling lambda and rho
    // initialize z and u and compute z-u and write to hadoop
    Map<String, LinearModel> z = new HashMap<String, LinearModel>(); // lambda ->
    List<String> lambdastr = props.getStringList(LAMBDA, ",");
    List<String> rhostr = props.getStringList(RHO, null, ",");
    if (rhostr != null)
    {
      if (rhostr.size() != lambdastr.size())
        throw new IOException("The number of rho's should be exactly the same as the number of lambda's. OR: don't claim rho!");
    }
    Map<Float, Float> lambdaRho = new HashMap<Float, Float>();
    for (int j = 0; j < lambdastr.size(); j++)
    {
      float lambda = Float.parseFloat(lambdastr.get(j));
      float rho;
      if (rhostr != null)
      {
        rho = Float.parseFloat(rhostr.get(j));
      }
      else
      {
        if (lambda <= 100)
        {
          rho = 1;
        }
        else
        {
          rho = 10;
        }
      }
      lambdaRho.put(lambda, rho);
      z.put(String.valueOf(lambda), new LinearModel());     
    }

    // Get specific lambda treatment for some features
    String lambdaMapPath = props.getString(LAMBDA_MAP, "");
    Map<String, Float> lambdaMap = new HashMap<String, Float>();
    if (!lambdaMapPath.equals(""))
    {
      AvroHdfsFileReader reader = new AvroHdfsFileReader(conf);
      ReadLambdaMapConsumer consumer = new ReadLambdaMapConsumer();
      reader.build(lambdaMapPath, consumer);
      consumer.done();
      lambdaMap = consumer.get();
    }
    _logger.info("Lambda Map has size = " + String.valueOf(lambdaMap.size()));
    // Write lambda_rho mapping into file
    String rhoPath = outBasePath + "/lambda-rho/part-r-00000.avro";
    writeLambdaRho(conf, rhoPath, lambdaRho);
    
    // test-loglik computation
    boolean testLoglikPerIter = props.getBoolean(TEST_LOGLIK_PER_ITER, false);
    DataFileWriter<GenericRecord> testRecordWriter = null;
    // test if the test file exists
    String testPath = props.getString(TEST_PATH, "");
    testLoglikPerIter = Util.checkPath(testPath);
    if (testLoglikPerIter)
    {
      List<Path> testPathList =
          AvroUtils.enumerateFiles(conf, new Path(testPath));
      if (testPathList.size() > 0)
      {
        testPath = testPathList.get(0).toString();
        _logger.info("Sample test path = " + testPath);
      
        AvroHdfsFileWriter<GenericRecord> writer =
            new AvroHdfsFileWriter<GenericRecord>(conf, outBasePath
                + "/sample-test-loglik/write-test-00000.avro", SampleTestLoglik.SCHEMA$);
        testRecordWriter = writer.get();
      }
    }
    if (testRecordWriter == null)
    {
      testLoglikPerIter = false;
      _logger.info("test.loglik.per.iter=false or test path doesn't exist or is empty! So we will not output test loglik per iteration.");
    }
    else
    {
      testRecordWriter.close();
    }
    
    MutableFloat bestTestLoglik = new MutableFloat(-9999999);
    //Initialize z by mean model 
    if (initializeBoostRate  > 0 && reg==2)
    {
      _logger.info("Now start mean model initializing......");
      // Different paths for L1 vs L2 set from job file
      String initalModelPath;
      initalModelPath = outBasePath + "/initialModel";

      Path initalModelPathFromNaiveTrain = new Path(outBasePath, "models");  
      JobConfig propsIni = JobConfig.clone(props);
      if (!propsIni.containsKey(LIBLINEAR_EPSILON))
      {
        propsIni.put(LIBLINEAR_EPSILON, 0.01);
      }
      propsIni.put(RegressionNaiveTrain.HEAVY_PER_ITEM_TRAIN, "true");
      propsIni.put(LAMBDA_MAP, lambdaMapPath);
      propsIni.put(REMOVE_TMP_DIR, "false");
      
      // run job
      RegressionNaiveTrain initializationJob = new RegressionNaiveTrain(super.getJobId()+"_ADMMInitialization",propsIni);
      initializationJob.run();
      
      FileSystem fs = initalModelPathFromNaiveTrain.getFileSystem(conf);
      if (fs.exists(new Path(initalModelPath)))
      {
        fs.delete(new Path(initalModelPath),true);
      }
      fs.rename(initalModelPathFromNaiveTrain, new Path(initalModelPath));
      // set up lambda
      Set<Float> lambdaSet = new HashSet<Float>();
      for (String l : lambdastr)
      {
        lambdaSet.add(Float.parseFloat(l));
      }
      // Compute Mean model as initial model
      z = LinearModelUtils.meanModel(conf, initalModelPath, nblocks, lambdaSet.size(), true);      

      if (testLoglikPerIter)
      {       
        updateLogLikBestModel(conf, 0,  z, testPath, ignoreValue, bestTestLoglik, outBasePath, numClickReplicates);
      }     
    }
    
    double mindiff = 99999999;
    float liblinearEpsilon = 0.01f;   
    int i;
    for (i = 1; i <= niter; i++)
    {
      _logger.info("Now starting iteration " + String.valueOf(i));
      // set up configuration
      props.put(AbstractAvroJob.OUTPUT_PATH, outBasePath + "/iter-" + String.valueOf(i));
      conf =
          createJobConf(AdmmMapper.class,
                        AdmmReducer.class,
                        Pair.getPairSchema(Schema.create(Type.INT),
                                           RegressionPrepareOutput.SCHEMA$),
                                           RegressionTrainOutput.SCHEMA$);
      conf.setPartitionerClass(AdmmPartitioner.class);
      //AvroUtils.setSpecificReducerInput(conf, true);
      conf.setInt(NUM_BLOCKS, nblocks);
      //Added for L1/L2
      conf.setInt(REGULARIZER, reg);
      conf.setLong(REPORT_FREQUENCY, props.getLong(REPORT_FREQUENCY, 1000000));
      //boolean ignoreValue = props.getBoolean(BINARY_FEATURE, false);
      conf.setBoolean(BINARY_FEATURE, ignoreValue);
      conf.setBoolean(SHORT_FEATURE_INDEX, props.getBoolean(SHORT_FEATURE_INDEX, false));

      boolean penalizeIntercept = props.getBoolean(PENALIZE_INTERCEPT, false);
      String interceptKey = props.getString(INTERCEPT_KEY, LibLinearDataset.INTERCEPT_NAME);
      conf.set(INTERCEPT_KEY, interceptKey);
      //int schemaType = props.getInt(SCHEMA_TYPE, 1);

      // compute and store u into file
      // u = uplusx - z
      String uPath = outBasePath + "/iter-" + String.valueOf(i) + "/u/part-r-00000.avro";
      if (i == 1)
      {
        LinearModelUtils.writeLinearModel(conf, uPath, new HashMap<String, LinearModel>());
        if (initializeBoostRate > 0 && reg==2)
        {
          
          conf.setFloat(RHO_ADAPT_RATE, initializeBoostRate);
        }
      }
      else
      {
          String uplusxPath = outBasePath + "/iter-" + String.valueOf(i - 1) + "/model";
          computeU(conf, uPath, uplusxPath, z);
        if(rhoAdaptCoefficient > 0)
        {
          float curRhoAdaptRate = (float) Math.exp(-(i-1)*rhoAdaptCoefficient);
          conf.setFloat(RHO_ADAPT_RATE, curRhoAdaptRate);
        }        
      }
      // write z into file
      String zPath = outBasePath + "/iter-" + String.valueOf(i) + "/init-value/part-r-00000.avro";
      LinearModelUtils.writeLinearModel(conf, zPath, z);

      // run job
      String outpath = outBasePath + "/iter-" + String.valueOf(i) + "/model";
      conf.set(U_PATH, uPath);
      conf.set(INIT_VALUE_PATH, zPath);
      conf.set(LAMBDA_RHO_MAP, rhoPath);
      if (i > 1 && mindiff < 0.001 && !aggressiveLiblinearEpsilonDecay) // need to get a more accurate estimate from liblinear
      {
        liblinearEpsilon = liblinearEpsilon / 10;
      }
      else if(aggressiveLiblinearEpsilonDecay && i > 5)
      {
          liblinearEpsilon = liblinearEpsilon / 10;
      }
      conf.setFloat(LIBLINEAR_EPSILON, liblinearEpsilon);
       //Added for logging aggressive decay
      _logger.info("Liblinear Epsilon for iter = " 
          + String.valueOf(i) + " is: " + String.valueOf(liblinearEpsilon));
      _logger.info("aggressiveLiblinearEpsilonDecay="+aggressiveLiblinearEpsilonDecay);
      AvroOutputFormat.setOutputPath(conf, new Path(outpath));
      AvroUtils.addAvroCacheFiles(conf, new Path(uPath));
      AvroUtils.addAvroCacheFiles(conf, new Path(zPath));
      AvroUtils.addAvroCacheFiles(conf, new Path(rhoPath));
      conf.setNumReduceTasks(nblocks * lambdastr.size());
      AvroJob.setInputSchema(conf, RegressionPrepareOutput.SCHEMA$);
      AvroUtils.runAvroJob(conf);
      // Load the result from the last iteration
      // compute z and u given x
      
      
      Map<String, LinearModel> xbar =
          LinearModelUtils.meanModel(conf, outpath, nblocks, lambdaRho.size(), true);
      Map<String, LinearModel> ubar = LinearModelUtils.meanModel(conf, uPath, nblocks, lambdaRho.size(), false);
      Map<String, LinearModel> lastz = new HashMap<String, LinearModel>();
      for (String k : z.keySet())
      {
        lastz.put(k, z.get(k).copy());
      }
      for (String lambda : xbar.keySet())
      {
        LinearModel thisz = z.get(lambda);
        thisz.clear();
        float l = Float.parseFloat(lambda);
        float r = lambdaRho.get(l);
        double weight;
        //L2 regularization
        if(reg==2) 
         {
            _logger.info("Running code for regularizer = " + String.valueOf(reg));
            weight = nblocks * r / (l + nblocks * r);
            Map<String, Double> weightmap = new HashMap<String, Double>();
              for (String k : lambdaMap.keySet())
              {
                 weightmap.put(k, nblocks * r / (lambdaMap.get(k) + nblocks * r + 0.0));
              }
              thisz.linearCombine(1.0, weight, xbar.get(lambda), weightmap);
              if (!ubar.isEmpty())
              {
                  thisz.linearCombine(1.0, weight, ubar.get(lambda), weightmap);
              }
              if (!penalizeIntercept)
              {
                 if (ubar.isEmpty())
                 {
                      thisz.setIntercept(xbar.get(lambda).getIntercept());
                 }
                 else
                 {
                      thisz.setIntercept(xbar.get(lambda).getIntercept()
                             + ubar.get(lambda).getIntercept());
                 }
              }
              z.put(lambda, thisz);
          }
          else
        {
          // L1 regularization

          _logger.info("Running code for regularizer = " + String.valueOf(reg));
          weight = l / (r * nblocks + 0.0);
          Map<String, Double> weightmap = new HashMap<String, Double>();
          for (String k : lambdaMap.keySet())
          {
            weightmap.put(k, lambdaMap.get(k) / (r * nblocks + 0.0));
          }
          // LinearModel thisz = new LinearModel();
          thisz.linearCombine(1.0, 1.0, xbar.get(lambda));
          if (!ubar.isEmpty())
          {
            thisz.linearCombine(1.0, 1.0, ubar.get(lambda));
          }
          // Iterative Thresholding
          Map<String, Double> thisCoefficients = thisz.getCoefficients();
          for (String k : thisCoefficients.keySet())
          {
            double val = thisCoefficients.get(k);
            if (val > weight)
            {
              thisCoefficients.put(k, val - weight);
            }
            else if (val < -weight)
            {
              thisCoefficients.put(k, val + weight);
            }
          }
          thisz.setCoefficients(thisCoefficients);
          if (!penalizeIntercept)
          {
            if (ubar.isEmpty())
            {
              thisz.setIntercept(xbar.get(lambda).getIntercept());
            }
            else
            {
              thisz.setIntercept(xbar.get(lambda).getIntercept()
                  + ubar.get(lambda).getIntercept());
            }
          }
          z.put(lambda, thisz);
        }
      }
      xbar.clear();
      ubar.clear();
      // Output max difference between last z and this z
      mindiff = 99999999;
      double maxdiff = 0;
      for (String k : z.keySet())
      {
        LinearModel tmp = lastz.get(k);
        if (tmp == null)
          tmp = new LinearModel();
        tmp.linearCombine(1, -1, z.get(k));
        double diff = tmp.maxAbsValue();
        _logger.info("For lambda=" + k + ": Max Difference between last z and this z = "
            + String.valueOf(diff));
        tmp.clear();
        if (mindiff > diff)
          mindiff = diff;
        if (maxdiff < diff)
          maxdiff = diff;
      }
      double epsilon = props.getDouble(EPSILON, 0.0001);
      // remove tmp files?
      if (props.getBoolean(REMOVE_TMP_DIR, false) && i >= 2)
      {
        FileSystem fs = FileSystem.get(conf);
        fs.delete(new Path(outBasePath + "/iter-" + String.valueOf(i - 1)), true);
      }
      // Output testloglik and update best model
      if (testLoglikPerIter)
      {
        updateLogLikBestModel(conf,
                              i,
                              z,
                              testPath,
                              ignoreValue,
                              bestTestLoglik,
                              outBasePath,
                              numClickReplicates);
      }

      if (maxdiff < epsilon && liblinearEpsilon <= 0.00001)
      {
        break;
      }
    }

    // write z into file
    String zPath = outBasePath + "/final-model/part-r-00000.avro";
    LinearModelUtils.writeLinearModel(conf, zPath, z);
    // remove tmp files?
    if (props.getBoolean(REMOVE_TMP_DIR, false))
    {
      FileSystem fs = FileSystem.get(conf);
      Path initalModelPath = new Path(outBasePath + "/initialModel");
      if (fs.exists(initalModelPath))
      {
        fs.delete(initalModelPath, true);
      }
      for (int j = i - 2; j <= i; j++)
      {
        Path deletepath = new Path(outBasePath + "/iter-" + String.valueOf(j));
        if (fs.exists(deletepath))
        {
          fs.delete(deletepath, true);
        }
      }
      fs.delete(new Path(outBasePath + "/tmp-data"), true);
    }

  }

  public static class AdmmMapper extends
  AvroMapper<RegressionPrepareOutput, Pair<Integer, RegressionPrepareOutput>>
  {
    private ReadLambdaRhoConsumer _lambdaRhoConsumer = new ReadLambdaRhoConsumer();

    @Override
    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      AvroDistributedCacheFileReader lambdaRhoReader =
          new AvroDistributedCacheFileReader(new JobConf(conf));
      try
      {
        lambdaRhoReader.build(conf.get(LAMBDA_RHO_MAP), _lambdaRhoConsumer);
        _lambdaRhoConsumer.done();
      }
      catch (IOException e)
      {
        e.printStackTrace();
      }
      _logger.info("lambda file:" + conf.get(LAMBDA_RHO_MAP));
      _logger.info("Loaded " + String.valueOf(_lambdaRhoConsumer.get().size())
                   + " lambdas.");
    }

    @Override
    public void map(RegressionPrepareOutput data,
                    AvroCollector<Pair<Integer, RegressionPrepareOutput>> collector,
                    Reporter reporter) throws IOException
    {
      Integer key = Integer.parseInt(data.key.toString());
      for (int i = 0; i < _lambdaRhoConsumer.get().size(); i++)
      {
        int newkey = key * _lambdaRhoConsumer.get().size() + i;
        // String newkey = String.valueOf(lambda)+"#"+key;
        data.key = String.valueOf(newkey);
        Pair<Integer, RegressionPrepareOutput> outPair =
            new Pair<Integer, RegressionPrepareOutput>(newkey, data);
        collector.collect(outPair);
      }
    }
  }

  public static class AdmmPartitioner implements
  Partitioner<AvroKey<Integer>, AvroValue<RegressionPrepareOutput>>
  {
    @Override
    public void configure(JobConf conf)
    {
    }

    @Override
    public int getPartition(AvroKey<Integer> key,
                            AvroValue<RegressionPrepareOutput> value,
                            int numPartitions)
    {
      Integer keyInt = key.datum();
      if (keyInt < 0 || keyInt >= numPartitions)
      {
        throw new RuntimeException("Map key is wrong! key has to be in the range of [0,numPartitions-1].");
      }
      return keyInt;
    }
  }

  public static class AdmmReducer extends
  AvroReducer<Integer, RegressionPrepareOutput, GenericData.Record>
  {
    String                        _interceptKey;
    long                          _reportfreq;
    boolean                       _binaryFeature;
    boolean                       _shortFeatureIndex;
    float                         _liblinearEpsilon;
    String                        _uPath;
    String                        _initValuePath;
    JobConf                       _conf;
    private ReadLambdaRhoConsumer _lambdaRhoConsumer = new ReadLambdaRhoConsumer();
    private List<Float>           _lambdaOrderedList;
    private float                _rhoAdaptRate;

    @Override
    public void setConf(Configuration conf)
    {
      super.setConf(conf);
      if (conf == null)
      {
        return;
      }
      _interceptKey = conf.get(INTERCEPT_KEY, LibLinearDataset.INTERCEPT_NAME);
      _reportfreq = conf.getLong(REPORT_FREQUENCY, 1000000);
      _binaryFeature = conf.getBoolean(BINARY_FEATURE, false);
      _shortFeatureIndex = conf.getBoolean(SHORT_FEATURE_INDEX, false);
      _liblinearEpsilon = conf.getFloat(LIBLINEAR_EPSILON, 0.01f);
      _rhoAdaptRate = conf.getFloat(RHO_ADAPT_RATE,  1.0f);
      AvroDistributedCacheFileReader lambdaRhoReader =
          new AvroDistributedCacheFileReader(new JobConf(conf));
      try
      {
        lambdaRhoReader.build(conf.get(LAMBDA_RHO_MAP), _lambdaRhoConsumer);
        _lambdaRhoConsumer.done();
      }
      catch (IOException e)
      {
        e.printStackTrace();
      }
      _uPath = conf.get(U_PATH);
      _initValuePath = conf.get(INIT_VALUE_PATH);
      _conf = new JobConf(conf);
      Set<Float> lambdaSet = _lambdaRhoConsumer.get().keySet();
      _lambdaOrderedList = new ArrayList<Float>(lambdaSet);
      java.util.Collections.sort(_lambdaOrderedList);
    }

    @Override
    public void reduce(Integer NumKey,
                       Iterable<RegressionPrepareOutput> values,
                       AvroCollector<GenericData.Record> collector,
                       Reporter reporter) throws IOException
    {
      int nlambdas = _lambdaRhoConsumer.get().size();
      float lambda = _lambdaOrderedList.get(NumKey % nlambdas);
      int partitionID = (int) NumKey / nlambdas;
      String key = String.valueOf(lambda) + "#" + String.valueOf(partitionID);
      // float lambda = Float.parseFloat(Util.getLambda(key.toString()));
      double rho = _lambdaRhoConsumer.get().get(lambda); 
      if (_rhoAdaptRate!=1.0)
      {
        rho = rho* (double) _rhoAdaptRate;
        _logger.info("Adaptive rate is " + _rhoAdaptRate);
        _logger.info("Adaptive rho is " + rho);
      }
     
      // get prior mean and init value
      FindLinearModelConsumer _uConsumer = new FindLinearModelConsumer(key.toString());
      FindLinearModelConsumer _initValueConsumer =
          new FindLinearModelConsumer(Util.getLambda(key.toString()));
      AvroDistributedCacheFileReader uReader = new AvroDistributedCacheFileReader(_conf);
      uReader.build(_uPath, _uConsumer);
      _uConsumer.done();
      _logger.info("Loaded u for the key, size:"
          + _uConsumer.get().getCoefficients().size());
      AvroDistributedCacheFileReader initValueReader =
          new AvroDistributedCacheFileReader(_conf);
      initValueReader.build(_initValuePath, _initValueConsumer);
      _initValueConsumer.done();
      _logger.info("Loaded initial value of the model, size:"
          + _initValueConsumer.get().getCoefficients().size());
      GenericData.Record output = new GenericData.Record(RegressionTrainOutput.SCHEMA$);
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
      for (RegressionPrepareOutput record : values)
      {
        dataset.addInstanceAvro(record);
      }
      dataset.finish();
      // Prepare the initial value
      LinearModel initvalue = _initValueConsumer.get();
      Map<String, Double> initvaluemap = initvalue.toMap(LibLinearDataset.INTERCEPT_NAME);
      // Prepare the prior mean
      LinearModel priormean = _uConsumer.get().copy();
      // Compute z minus u
      priormean.linearCombine(-1, 1, initvalue);
      Map<String, Double> priormeanmap = priormean.toMap(LibLinearDataset.INTERCEPT_NAME);
      // Run liblinear
      LibLinear liblinear = new LibLinear();
      liblinear.setReporter(reporter, _reportfreq);
      String option = "epsilon=" + String.valueOf(_liblinearEpsilon);
      try
      {
        liblinear.train(dataset, initvaluemap, priormeanmap, null, 1.0 / rho, option);
        LinearModel model = liblinear.getLinearModel();
        output.put("key", key);
        output.put("model", model.toAvro(LibLinearDataset.INTERCEPT_NAME));
        LinearModel uplusx = _uConsumer.get();
        uplusx.linearCombine(1, 1, model);
        output.put("uplusx", uplusx.toAvro(LibLinearDataset.INTERCEPT_NAME));
      }
      catch (Exception e)
      {
        throw new IOException("Model fitting error!", e);
      }
      collector.collect(output);
    }
  }

  private void writeLambdaRho(JobConf conf, String path, Map<Float, Float> lambda_rho) throws IOException
  {
    AvroHdfsFileWriter<GenericRecord> writer =
        new AvroHdfsFileWriter<GenericRecord>(conf, path, LambdaRhoMap.SCHEMA$);
    DataFileWriter<GenericRecord> recordWriter = writer.get();
    for (Float k : lambda_rho.keySet())
    {
      GenericRecord record = new GenericData.Record(LambdaRhoMap.SCHEMA$);
      record.put("lambda", k);
      record.put("rho", lambda_rho.get(k));
      recordWriter.append(record);
    }
    recordWriter.close();
  }
  // u = u + x - z
  private void computeU(JobConf conf, String uPath, String uplusxPath, Map<String, LinearModel> z) throws IOException
  {
    AvroHdfsFileWriter<GenericRecord> writer =
        new AvroHdfsFileWriter<GenericRecord>(conf, uPath, LinearModelAvro.SCHEMA$);
    DataFileWriter<GenericRecord> recordwriter = writer.get();
    // read u+x
    for (Path path : Util.findPartFiles(conf, new Path(uplusxPath)))
    {
      DataFileStream<Object> stream = AvroUtils.getAvroDataStream(conf, path);
      while (stream.hasNext())
      {
        GenericData.Record record = (GenericData.Record) stream.next();
        String partitionID = Util.getStringAvro(record, "key", false);
        if (record.get("uplusx") != null)
        {
          String lambda = Util.getLambda(partitionID);
          LinearModel newu =
              new LinearModel(LibLinearDataset.INTERCEPT_NAME, (List<?>) record.get("uplusx"));
          newu.linearCombine(1.0, -1.0, z.get(lambda));
          GenericData.Record newvaluemap =
              new GenericData.Record(LinearModelAvro.SCHEMA$);
          List modellist = newu.toAvro(LibLinearDataset.INTERCEPT_NAME);
          newvaluemap.put("key", partitionID);
          newvaluemap.put("model", modellist);
          recordwriter.append(newvaluemap);
        }
      }
    }
    recordwriter.close();
  }
  private Map<String, Double> testloglik(JobConf conf, Map<String, LinearModel> modelmap,
                                         String testPath,
                                         int num_click_replicates,
                                         boolean ignore_value) throws IOException
  {
    DataFileStream<Object> stream = AvroUtils.getAvroDataStream(conf, new Path(testPath));
    Map<String, Double> loglik = new HashMap<String, Double>();
    for (String k : modelmap.keySet())
    {
      loglik.put(k, 0.0);
    }
    double n = 0;
    long nrecords = 0;
    while (stream.hasNext())
    {
      GenericData.Record record = (GenericData.Record) stream.next();
      for (String k : modelmap.keySet())
      {
        double tmp = loglik.get(k);
        loglik.put(k,
                   tmp
                   + modelmap.get(k).evalInstanceAvro(record,
                                                      true,
                                                      num_click_replicates,
                                                      ignore_value));
      }
      double weight = 1;
      if (record.get("weight")!=null)
      {
        weight = Double.parseDouble(record.get("weight").toString());
      }
      nrecords++;
      n += weight;
      if (nrecords >= MAX_NTEST_EVENTS)
      {
        break;
      }
    }
    for (String k : loglik.keySet())
    {
      double tmp = loglik.get(k);
      loglik.put(k, tmp / n);
    }
    _logger.info("Finished computing testloglik...Evaluated #test records=" + nrecords);
    return loglik;
  }
  private void updateLogLikBestModel(JobConf conf, int niter,  Map<String, LinearModel> z, String testPath, 
                                     boolean ignoreValue, MutableFloat bestTestLoglik, String outBasePath, 
                                     int  numClickReplicates) throws IOException
   {   
     Map<String, Double> loglik;
     loglik = testloglik(conf, z, testPath, 1, ignoreValue);
     
     AvroHdfsFileWriter<GenericRecord> writer =
         new AvroHdfsFileWriter<GenericRecord>(conf, outBasePath
             + "/sample-test-loglik/iteration-"+niter +".avro", SampleTestLoglik.SCHEMA$);
     DataFileWriter<GenericRecord> testRecordWriter = writer.get();  

     for (String k : z.keySet())
     {     
       GenericData.Record valuemap = new GenericData.Record(SampleTestLoglik.SCHEMA$);
       valuemap.put("iter", niter);
       valuemap.put("testLoglik", loglik.get(k).floatValue());
       valuemap.put("lambda", k);
       testRecordWriter.append(valuemap);
       _logger.info("Sample test loglik for lambda=" + k + " is: "
           + String.valueOf(loglik.get(k)));
      
       // output best model up to now
       if (loglik.get(k) > bestTestLoglik.floatValue() && niter>0)
       {
         String bestModelPath = outBasePath + "/best-model/best-iteration-" + niter + ".avro";
         FileSystem fs = FileSystem.get(conf);
         fs.delete(new Path(outBasePath + "/best-model"), true);
         LinearModelUtils.writeLinearModel(conf, bestModelPath, z.get(k), k);
         bestTestLoglik.setValue(loglik.get(k).floatValue());
       }
     }
     testRecordWriter.close();
   }   
}
