ml-ease


Table of Contents
=================

- What is ml-ease?
- Copyright
- Open Source Software included in ml-ease
- What is ADMM?
- Quick Start
- Code structure
- Input Data Format
- Output Models
- Detailed Instructions
- Supporting Team


What is ml-ease?
=============

ml-ease is the Open-sourced Large-scale machine learning library from LinkedIn; currently it has ADMM based large scale logistic regression.

Copyright
=========

[2014] LinkedIn Corp. All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");?you may not use this file except in compliance with the License.?You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software?distributed under the License is distributed on an "AS IS" BASIS,?WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 
Open Source Software included in ml-ease
=====================================

- liblinear
   https://github.com/bwaldvogel/liblinear-java
   Copyright (c) 2007-2013 The LIBLINEAR Project.
   License: Apache 2.0 (see below)

What is ADMM?
=============

ADMM stands for Alternating Direction Method of Multipliers (Boyd et al. 2011). The basic idea of ADMM is as follows: ADMM considers the large scale logistic regression model fitting as a convex optimization problem with constraints. The ADMM algorithm is guaranteed to converge.? While minimizing the user-defined loss function, it enforces an extra constraint that coefficients from all partitions have to equal. To solve this optimization problem, ADMM uses an iterative process. For each iteration it partitions the big data into many small partitions, and fit an independent logistic regression for each partition. Then, it aggregates the coefficients collected from all partitions, learns the consensus coefficients, and sends it back to all partitions to retrain. After 10-20 iterations, it ends up with a converged solution that is theoretically close to what you would have obtained if you trained it on a single machine.

Quick Start
===========

- Installation: This uses maven to compile. Command: "mvn clean install"
- Run ADMM: 
  1. Copy the jar ./target/ml-ease-1.0-jar-with-dependencies.jar to hadoop gateway. And then set up a config file like ./examples/sample-config.job.
  2.To run Admm: hadoop jar ml-ease-1.0-jar-with-dependencies.jar com.linkedin.ml-ease.regression.jobs.Regression sample-config.job.

Code structure
==============

- You can see under src/main/java there are two directories: avro and bjson. Naturally, avro dir has the code for data in avro format.
- Inside the avro directory you will see 10-15 java classes. It might look a little overwhelming in the beginning, but the most important two classes are:
  1. AdmmPrepare.java is the job you need to call before you run ADMM (i.e. AdmmTrain.java). This job makes sanity checks of the input data format, and assigns partition id to each sample.
  2. AdmmTrain.java is the job you need to call for training logistic regression with L2. It allows you to specify multiple L2 penalty parameters (i.e. lambda) at one run, and if you specify the test data, it will grab the first 1M samples or the first partition in your test data path, and outputs log-likelihood for each iteration and each lambda value.
- Other useful hadoop jobs include:
  1. AdmmTest.java is a job for testing the model trained by AdmmTrain.java on test data.
  2. AdmmTestLoglik.java is a job for computing loglikelihood given the output from AdmmTest.java.
  3. NaiveTrain.java is a job for training independent logistic regressions per key (This is very useful when you want to train say per-item model on hadoop while data for each item is quite small).
- A standard flow for training and test ADMM is: AdmmPrepare -> AdmmTrain -> AdmmTest -> AdmmTestLoglik.
Input Data Format
- The data must have two fields:
- response:int,
 features:[{name:string, term:string, value:float}]
All the other fields are optional.
- We define a feature string to be represented as name + term. For example, suppose you have a feature called age=[10,20], i.e. age between 10 years old to 20 years old. The feature can be represented by:
  name="age"
  term="[10,20]"
  value=1.0
- If you don't want to use two fields to represent one feature, feel free to set term as empty string. Note that name should never be empty strings.
- Training and Test data should have the same format. If not, it is probably fine as long as both have fields "response" and "features".
- Intercept does not need to be put into the training data.
- Below is a sample of training/test data:
  Record 1:
  {
    "response" : 0,
    "features" : [ {
      "name" : "7",
      "term" : "33",
      "value" : 1.0
    }, {
      "name" : "8",
      "term" : "151",
     "value" : 1.0
    }, {
      "name" : "3",
      "term" : "0",
      "value" : 1.0
    }, {
      "name" : "12",
      "term" : "132",
      "value" : 1.0
    } ],
    "weight" : 1.0,
    "offset" : 0.0,
    "foo" : "whatever"
 }
- Weight is an optional field that specifies the weight of the observation. Default is 1.0. If you feel some observation is stronger than the others, feel free to use this field, say making the weak ones 0.5.
- Offset is an optional field. Default is 0.0. When it is non-zero, the model will learn coefficients beta by x'beta+offset instead of x'beta.
- foo is an extra field to let you know that ADMM allows you to put extra fields in.


Output Models
=============

- Output Directory Structure
- In the output model directory, there are several subdirs:
  1. "best-model" saves the best model based on the sample test data over all iterations for all lambdas. Use it WISELY because it may not mean the real "best-model". For example, if your sample test data is too small, the variance of the model performance will become high so that the best model doesn't mean anything. In that case use models from "final-model" instead.
  2. "final-model" saves the models for EACH lambda after the last iteration. This is always a safe choice and it is strongly recommended.
  3. "sample-test-loglik" saves the sample test loglikelihood over all iterations for all lambdas. This is simply for you to study the convergence.
- Output Model Format
A model output record has the following format:
key:string,
 model:[{name:string, term:string, value:float}]
 "key" column saves the lambda value.
 "model" saves the model output for that lambda. "name" + "term" again represents a feature string, and "value" is the learned coefficient. Note that the first element of the model is always "(INTERCEPT)", which means the intercept.     Below is a sample of the learned model for lambda = 1.0 and 2.0:
 Record 1:
  {
    "key" : "1.0",
    "model" : [ {
      "name" : "(INTERCEPT)",
      "term" : "",
      "value" : -2.5
    }, {
      "name" : "7",
      "term" : "33",
      "value" : 0.98
    }, {
      "name" : "8",
      "term" : "151",
      "value" : 0.34
    }, {
      "name" : "3",
      "term" : "0",
      "value" : -0.4
    }, {
      "name" : "12",
      "term" : "132",
      "value" : -0.3
    } ],
  }
  Record 2:
  {
    "key" : "2.0",
    "model" : [ {
      "name" : "(INTERCEPT)",
      "term" : "",
      "value" : -2.5
    }, {
      "name" : "7",
      "term" : "33",
      "value" : 0.83
    }, {
      "name" : "8",
      "term" : "151",
      "value" : 0.32
    }, {
      "name" : "3",
      "term" : "0",
      "value" : -0.3
    }, {
      "name" : "12",
      "term" : "132",
      "value" : -0.1
    } ],
 }

Detailed Instructions
=====================

- AdmmPrepare Job
input.pathsThe input path of training dataoutput.pathThe ROOT output path of the model directorynum.blocksNumber of partitions for ADMM, choose a large enough value so that your memory doesn't blow up. 
But note: The smaller this value is, the better convergence becomes.binary.featureAre all the features in this data binary?true/false
map.keyThe field name for partitioning the data. This is only useful for training per-item model using NaiveTrain.java"campaign_id"

num.click.replicatesThis is a trick: For sparse data sets where positives are rare, we replicate the clicks into N copies and give each of them 1/N weight. 
This helps in terms of convergence, but also makes the data larger. But note: it should be less than number of partitions.1 when num.blocks<10. 
5 when num.blocks>10.
- AdmmTrain Job
 It shares parameters with AdmmPrepare job, e.g. num.blocks, binary.feature, num.click.replicates. Please make sure they are the same as what are specified in AdmmPrepare.job.

- Other parameters include:
input.pathsOutput path of the Admm Prepare jobADMM-Prepare.output.path/tmp-data
output.model.pathThe ROOT output path of the model directoryADMM-Prepare.output.path
test.pathThe test data pathhas.interceptWhether the model has the intercept or not, if not, intercept will be 0true/false
lambdaL2 Penalty parameters1,10,100
num.itersnumber of ADMM iterations20
remove.tmp.dirWhether to remove tmp directories or not?true/false
epsilonConvergence parameter of ADMM0.0001
lambda.mapLocation of the lambda map on hdfs. This is for specifying different L2 penalty parameters for different coefficients. No need to use it in most casesshort.feature.indexHow many features do you have? If the number is less than short.MAX, then set to be true, otherwise falsetrue/false
test.loglik.per.iterOutput test logliklihood per iteration? Usually setting to be true is goodtrue/false
- AdmmTest Job
input.pathsThe test data pathoutput.base.pathThe ROOT path of output for test resultsmodel.base.pathThe ROOT path of the model outputADMM-Prepare.output.path
- AdmmTestLoglik Job
 This job will put a /_loglik subdir inside each test predicted directory.
input.base.paths
The ROOT path of output for test results
Admm-Test.output.path
output.base.pathThe ROOT path of output for test-loglik results
Admm-Test.output.path
- NaiveTrain job
This job is mainly for training per-item model, i.e. for each item (such as campaign_id, creative_id) train an independent regression model.
It can also be used for training one logistic regression model for a large scale data. That's why it is called "naive" train: It splits the data into partitions, train independent regression models for each partition, and then take average of the coefficients.
 The job parameters are very similar to AdmmTrain.java, except

compute.model.meanWhether to compute the mean of the coefficients that are learned from each partition. 
Claim it to be true only if you are training one regression model for a large data using naive method.true/false
data.size.thresholdFor per-item model, whether to ignore the item if the data size of this item is smaller than the threshold.Supporting Team
=================

This tool is developed by Applied Relevance Science team at LinkedIn. People who contributed to this tool include:
* Deepak Agarwal
* Bee-Chung Chen
* Bo Long
* Liang Zhang

