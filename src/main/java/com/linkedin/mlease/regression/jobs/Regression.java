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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.log4j.Logger;

import com.linkedin.mapred.AbstractAvroJob;
import com.linkedin.mapred.JobConfig;

public class Regression extends AbstractAvroJob
{
  private static final Logger _log = Logger.getLogger(Regression.class);
  public static final String OUTPUT_BASE_PATH        = "output.base.path";
  public static final String TEST_PATH               = "test.path";
  public Regression(String jobId, JobConfig config)
  {
    super(jobId, config);
  }

  @Override
  public void run() throws Exception
  {
    JobConfig config = super.getJobConfig();
    Path outBasePath = new Path(config.get(OUTPUT_BASE_PATH));
    JobConf conf = super.createJobConf();
    if (config.getBoolean("force.output.overwrite", false))
    {
      FileSystem fs = outBasePath.getFileSystem(conf);
      fs.delete(outBasePath, true);
    }
    
    String prepareOutputPath = outBasePath + "/tmp-data";
    // first run the preparation job
    JobConfig configPrepare = JobConfig.clone(config);
    configPrepare.put(AbstractAvroJob.OUTPUT_PATH, prepareOutputPath);
    RegressionPrepare regressionPrepareJob = new RegressionPrepare("Regression-Prepare", 
                                                                   configPrepare);
    regressionPrepareJob.run();
    
    // now start running the regression train using admm
    JobConfig configTrain = JobConfig.clone(config);
    configTrain.put(AbstractAvroJob.INPUT_PATHS, prepareOutputPath);
    RegressionAdmmTrain regressionAdmmTrainJob = new RegressionAdmmTrain("Regression-Admm-Train", configTrain);
    regressionAdmmTrainJob.run();
    
    // now test
    if (config.containsKey(TEST_PATH))
    {
      JobConfig configTest = JobConfig.clone(config);
      configTest.put(AbstractAvroJob.INPUT_PATHS, config.get(TEST_PATH));
      configTest.put(RegressionTest.MODEL_BASE_PATH, outBasePath.toString());
      String outTestBasePath = outBasePath.toString()+"/test";
      configTest.put(RegressionTest.OUTPUT_BASE_PATH, outTestBasePath);
      RegressionTest regressionTestJob = new RegressionTest("Regression-Test", configTest);
      regressionTestJob.run();
      
      // compute test loglikelihood
      JobConfig configTestLoglik = JobConfig.clone(config);
      configTestLoglik.put(RegressionTestLoglik.INPUT_BASE_PATHS, outTestBasePath);
      configTestLoglik.put(RegressionTestLoglik.OUTPUT_BASE_PATH, outTestBasePath);
      RegressionTestLoglik regressionTestLoglikJob = new RegressionTestLoglik("Regression-Test-Loglik", configTestLoglik);
      regressionTestLoglikJob.run();
    }
  }
 
  /**
   * The main function for logistic regression using ADMM
   * It contains only one argument: the path of the job config file
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception
  {
    if (args.length<1)
    {
      _log.error("[Usage]: Regression <Job config path>");
      return;
    }
    JobConfig config = new JobConfig(args[0]);
    Regression regression = new Regression("Regression-ADMM", config);
    regression.run();
  }

}
