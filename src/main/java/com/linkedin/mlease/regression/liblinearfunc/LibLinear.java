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

package com.linkedin.mlease.regression.liblinearfunc;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.hadoop.mapred.Reporter;

import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.regression.liblinearfunc.LibLinearFunction;
import com.linkedin.mlease.regression.liblinearfunc.LogisticRegressionL2;
import com.linkedin.mlease.regression.liblinearfunc.LogisticRegressionL2BinaryFeature;
import com.linkedin.mlease.utils.Util;

import de.bwaldvogel.liblinear.Tron;

/**
 * <p>
 * Fit a model using LibLinear
 * </p>
 * <p>
 * Example:
 * </p>
 * 
 * <pre>
 * LibLinearDataset dataset = new LibLinearDataset(1.0); // add an intercept/bias
 * // Or,                   = new LibLinearBinaryDataset(1.0, false); // to save memory
 * 
 * // add records into the dataset
 * for(...){
 *     ...
 *     dataset.addInstanceJSON(record); // record is a Map of JSON object
 * }
 * dataset.finish(); // finish() must be called before training
 * LibLinear liblinear = new LibLinear();
 * liblinear.train(dataset, 1.0, "max_iter=50, epsilon=0.01, positive_weight=5");
 * LinearModel model = liblinear.getLinearModel();
 * </pre>
 * <p>
 * Options:
 * </p>
 * <ul>
 * <li>max_iter: maximum number of iterations</li>
 * <li>epsilon: precision of the solution</li>
 * <li>positive_weight: weight to be added to the positive instances</li>
 * </ul>
 * 
 * IMPORTANT NOTE: voldemort.serialization.json.JsonReader seems to have a bug
 * when reading double values.  Use string instead.  For example, 
 * {... "offset"="-1.23", ...}, instead of {... "offset"=-1.23, ...}
 * 
 * @author bchen
 */
public class LibLinear
{
  // fitted parameter vector (in the map format)
  Map<String, Double>  coeff              = null;
  // fitted parameter vector (in the array format)
  double[]             param              = null;

  // posterior variance (only diagonal elements)
  Map<String, Double>  postVarMap         = null; // in the map format
  double[]             postVar            = null; // in the array format
  
  double[]             priorMean          = null;
  double[]             priorVar           = null;

  public double        bias               = 0;

  double               epsilon            = 0.01;
  String               type               = Logistic_L2_primal;
  int                  max_iter           = 10000;
  int                  verbose            = 0;
  double               positive_weight    = 1;

  public static String Logistic_L2_primal = "Logistic_L2_primal";
  public static String Do_nothing         = "Do_nothing";

  Reporter             reporter           = null;
  long                 reportFrequency    = 10000000;

  boolean              computeFullPostVar = false;
  double[][]           postVarMatrix      = null;
  Map<List<String>, Double> postVarMatrixMap = null; // key: [FeatureName1, FeatureName2]
                                                     // value: covariance
                                                     // including FeatureName1==FeatureName2
  
  void parseOption(String option) throws Exception
  {
    if (option == null)
      return;
    if ("".equals(option))
      return;
    String[] token = option.split("\\s*,\\s*");
    for (int i = 0; i < token.length; i++)
    {
      String[] pair = token[i].split("\\s*=\\s*");
      if (pair.length != 2)
        throw new Exception("Unknown option specification: '" + token[i] + "' in '"
            + option + "'");
      try
      {
        if (pair[0].equals("epsilon"))
        {
          epsilon = Util.atof(pair[1]);
        }
        else if (pair[0].equals("type"))
        {
          type = pair[1];
        }
        else if (pair[0].equals("max_iter"))
        {
          max_iter = Util.atoi(pair[1]);
        }
        else if (pair[0].equals("verbose"))
        {
          verbose = Util.atoi(pair[1]);
        }
        else if (pair[0].equals("positive_weight"))
        {
          positive_weight = Util.atof(pair[1]);
        }
        else
          throw new Exception();
      }
      catch (Exception e)
      {
        throw new Exception("Invalid option specification: '" + token[i] + "' in '"
            + option + "'");
      }
    }
  }

  /**
   * Whether to compute the full posterior variance-covariance matrix
   * @param compute
   */
  public void setComputeFullPostVar(boolean compute)
  {
    computeFullPostVar = compute;
  }
  
  /**
   * Set the reporter to report progress after processing n instances
   * 
   * @param rep
   * @param n
   */
  public void setReporter(Reporter rep, long n)
  {
    reporter = rep;
    if (n < 1000)
      n = 1000;
    reportFrequency = n;
  }

  /**
   * Train (i.e., fit or build) a model using input dataset according to the options.
   * 
   * @param dataset
   *          Training dataset
   * @param initParam
   *          Initial parameter vector (can be null, meaning all 0)
   * @param priorMean
   *          The prior mean of the parameter vector (can be null, meaning all 0)
   * @param priorVar
   *          The prior variance of the parameter vector (can be null, meaning all prior
   *          variances = priorVar_rest)
   * @param defaultPriorVar
   *          The prior variance for the parameters not specified in priorVar.
   * @param option
   *          A comma-separated list of options
   * @throws Exception
   */
  public void train(LibLinearDataset dataset,
                    Map<String, Double> initParam,
                    Map<String, Double> priorMean,
                    Map<String, Double> priorVar,
                    double defaultPriorVar,
                    String option) throws Exception
  {
    train(dataset, initParam, priorMean, priorVar, 0.0, defaultPriorVar, option);
  }

  public void train(LibLinearDataset dataset,
                    Map<String, Double> initParam,
                    Map<String, Double> priorMean,
                    Map<String, Double> priorVar,
                    double defaultPriorMean,
                    double defaultPriorVar,
                    String option) throws Exception
  {
    train(dataset, initParam, priorMean, priorVar, defaultPriorMean, defaultPriorVar, option, false);
  }
  
  public void train(LibLinearDataset dataset,
                    Map<String, Double> initParam,
                    Map<String, Double> priorMean,
                    Map<String, Double> priorVar,
                    double defaultPriorMean,
                    double defaultPriorVar,
                    String option,
                    boolean computePosteriorVar) throws Exception
  {
    if (!dataset.isFinished())
      throw new IOException("Cannot train a model using unfinished dataset");
    bias = dataset.bias;
    parseOption(option);

    // setup initial parameter vector
    param = new double[dataset.nFeatures()];
    initSetup(param, initParam, dataset, 0.0);

    // setup prior mean
    this.priorMean = new double[dataset.nFeatures()];
    initSetup(this.priorMean, priorMean, dataset, defaultPriorMean);

    // setup prior var
    this.priorVar = new double[dataset.nFeatures()];
    initSetup(this.priorVar, priorVar, dataset, defaultPriorVar);

    // setup initial posterior variance
    postVar = null; postVarMap = null;
    postVarMatrix = null; postVarMatrixMap = null;
    if(computePosteriorVar)
    {
      // initialize the diagonal posterior variance
      postVar = new double[this.priorVar.length];
      for(int i=0; i<postVar.length; i++) postVar[i] = this.priorVar[i];
      
      if(computeFullPostVar)
      {
        // initialize the full posterior variance matrix
        postVarMatrix = new double[postVar.length][];
        for(int i=0; i<postVarMatrix.length; i++)
        {
          postVarMatrix[i] = new double[postVarMatrix.length];
          for(int j=0; j<postVarMatrix.length; j++)
          {
            if(i==j) postVarMatrix[i][j] = this.priorVar[i];
            else     postVarMatrix[i][j] = 0;
          }
        }
      }
    }

    int pos = 0;
    for (int i = 0; i < dataset.nInstances(); i++)
      if (dataset.y[i] == 1)
        pos++;
    int neg = dataset.nInstances() - pos;

    if (type.equals(Logistic_L2_primal))
    {
      double multiplier = 1;
      LibLinearFunction func;
      
      if (dataset instanceof LibLinearBinaryDataset)
      {
        func =
            new LogisticRegressionL2BinaryFeature((LibLinearBinaryDataset) dataset,
                                                  this.priorMean,
                                                  this.priorVar,
                                                  multiplier,
                                                  positive_weight,
                                                  1);
      }
      else
      {
        func =
            new LogisticRegressionL2(dataset,
                                     this.priorMean,
                                     this.priorVar,
                                     multiplier,
                                     positive_weight,
                                     1);
      }
      if (reporter != null)
      {
        reporter.setStatus("Start LibLinear of type " + type);
        func.setReporter(reporter, reportFrequency);
      }
      
      // Find the posterior mode
      Tron tron =
          new Tron(func, epsilon * Math.min(pos, neg) / dataset.nInstances(), max_iter);
      tron.tron(param);
      
      // Compute the posterior variance
      if(computePosteriorVar)
      {
        if(computeFullPostVar)
        {
          // Compute the full posterior variance matrix
          func.hessian(param, postVarMatrix);
          RealMatrix H = MatrixUtils.createRealMatrix(postVarMatrix);
          CholeskyDecomposition decomp = new CholeskyDecomposition(H);
          DecompositionSolver solver = decomp.getSolver();
          RealMatrix Var = solver.getInverse();
          postVarMatrix = Var.getData();
          for(int i=0; i<postVar.length; i++) postVar[i] = postVarMatrix[i][i];
        }
        else
        {
          // Compute the diagonal elements of the variance
          func.hessianDiagonal(param, postVar);
          for(int i=0; i<postVar.length; i++) postVar[i] = 1.0/postVar[i];
        }
      }
    }
    else if (type.equals(Do_nothing))
    {
      // do nothing
    }
    else
      throw new Exception("Unknown type: " + type);

    coeff = new HashMap<String, Double>();
    if(computePosteriorVar) postVarMap = new HashMap<String,Double>();
    for (int index = 1; index <= dataset.nFeatures(); index++)
    {
      String featureName = dataset.getFeatureName(index);
      coeff.put(featureName, param[index - 1]);
      if(computePosteriorVar) postVarMap.put(featureName, postVar[index - 1]);
    }
    
    if(computePosteriorVar && computeFullPostVar)
    {
      postVarMatrixMap = new HashMap<List<String>, Double>();
      for (int i = 1; i <= dataset.nFeatures(); i++)
      {
        String name_i = dataset.getFeatureName(i);
        for (int j = 1; j <= dataset.nFeatures(); j++)
        {
          double cov = postVarMatrix[i-1][j-1];
          if(cov != 0)
          {
            String name_j = dataset.getFeatureName(j);
            ArrayList<String> pair = new ArrayList<String>(2);
            pair.add(name_i);
            pair.add(name_j);
            postVarMatrixMap.put(pair, cov);
          }
        }
      }
    }
    
    // check for features with non-zero prior that do not appear in the dataset
    if (priorMean != null)
    {
      for (String key : priorMean.keySet())
      {
        if (!coeff.containsKey(key))
        {
          coeff.put(key, priorMean.get(key));
        }
      }
    }
    if (priorVar != null && computePosteriorVar)
    {
      for (String key : priorVar.keySet())
      {
        if(!postVarMap.containsKey(key)) postVarMap.put(key, priorVar.get(key));
        if(computeFullPostVar)
        {
          ArrayList<String> pair = new ArrayList<String>(2);
          pair.add(key);
          pair.add(key);
          if(!postVarMatrixMap.containsKey(pair)) postVarMatrixMap.put(pair, priorVar.get(key));
        }
      }
    }
  }
  
  /**
   * Same as train(dataset, null, null, null, priorVar, option)
   * 
   * @param dataset
   * @param priorVar
   * @param option
   * @throws Exception
   */
  public void train(LibLinearDataset dataset, double priorVar, String option) throws Exception
  {
    train(dataset, null, null, null, priorVar, option);
  }

  /**
   * Get the regression coefficients as a map
   * @return a map where the key is a feature name and the value is the
   *         posterior mean of the corresponding regression coefficient
   */
  public Map<String, Double> getParamMap()
  {
    return coeff;
  }

  /**
   * Get the posterior variances of regression coefficients as a map.
   * This method only returns the diagonal elements of the 
   * variance-covariance matrix.
   * It returns null if computePosteriorVar==false when calling the
   * train method.
   * @return a map where the key is a feature name and the value is the
   *         posterior variance of the corresponding regression coefficient
   */
  public Map<String, Double> getPostVarMap()
  {
    return postVarMap;
  }
  
  /**
   * Get the full posterior variance-covariance matrix of regression coefficients
   * as a map.
   * It returns null if computePosteriorVar==false when calling the
   * train method or computeFullPostVar==false.
   * <b>Nonexistence of a pair of features means zero posterior correlation
   * between the corresponding two regression coefficients.</b>
   * @return a map where the key is a pair of feature names and the value is the
   *         posterior covariance of the corresponding two regression coefficients.
   *         The pair of feature names can be the same; in this case, the value
   *         is the posterior variance.
   */
  public Map<List<String>, Double> getPostVarMatrixMap()
  {
    return postVarMatrixMap;
  }
  
  /**
   * Return the fitted linear model
   * 
   * @return
   * @throws Exception
   */
  public LinearModel getLinearModel() throws Exception
  {
    if (coeff == null)
      throw new Exception("This model has not been built.  Please call train() before calling getLinearModel().");
    LinearModel model = null;
    if (bias > 0)
    {
      model = new LinearModel(LibLinearDataset.INTERCEPT_NAME, coeff);
    }
    else
    {
      model = new LinearModel(0.0, coeff);
    }
    return model;
  }

  void initSetup(double[] param,
                 Map<String, Double> map,
                 LibLinearDataset dataset,
                 double defaultValue)
  {
    for (int i = 0; i < param.length; i++)
      param[i] = defaultValue;
    if (map != null)
    {
      Iterator<Entry<String, Double>> iter = map.entrySet().iterator();
      while (iter.hasNext())
      {
        Entry<String, Double> entry = iter.next();
        if (entry.getKey() == null)
          throw new RuntimeException("input key is null");
        int index = dataset.getFeatureIndex(entry.getKey());
        if (index == -1)
          continue; // the feature name does not appear in the dataset
        param[index - 1] = entry.getValue(); // index starts from 1
      }
    }
  }

  static final String RUN_TRAIN   = "train";
  static final String RUN_PREDICT = "predict";

  static void cmd_line_error(String msg, String cmd)
  {
    System.err.println("\nERROR: " + msg);
    System.err.println("\n" + cmd);
    System.exit(0);
  }

  /**
   * Command-line tool
   * 
   * <pre>
   * java -cp target/regression-0.1-uber.jar com.linkedin.lab.regression.LibLinear
   * </pre>
   * 
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception
  {

    String cmd =
        "Input parameters (separated by space): \n"
            + "   run:<command>      (required) train or predict\n"
            + "   ftype:<file_type>  (required) libsvm or json\n"
            + "   data:<file_name>   (required) Input data file of the specified type\n"
            + "   out:<file_name>    (required) Output file\n"
            + "   bias:<bias>        (optional) Set to 0 if you do not want to add an\n"
            + "                                 bias/intercept term\n"
            + "                                 Set to 1 if you want to add a feature with\n"
            + "                                 value 1 to every instance\n"
            + "                                 Default: 0\n"
            + "   param:<file_name>  (optional) for run:train, it specifies the prior mean\n"
            + "                      (required) for run:predict, it specifies the model\n"
            + "                                 File format: <featureName>=<value> per line\n"
            + "   priorVar:<var>     (required) for run:train, <var> is the a number\n"
            + "                      (not used) for run:predict\n"
            + "   init:<file_name>   (optional) for run:train, it specifies the initial value\n"
            + "                                 File format: <featureName>=<value> per line\n"
            + "   posteriorVar:1/0   (optional) Whether to compute posterior variances\n"
            + "                                 Default: 1\n"
            + "   posteriorCov:1/0   (optional) Whether to compute posterior covariances\n"
            + "                                 Default: 0\n"
            + "   binaryFeature:1/0  (optional) Whether all of the input features are binary\n"
            + "   useShort:1/0       (optional) Whether to use short to store feature indices\n"
            + "   option:<options>   (optional) Comma-separated list of options\n"
            + "                                 No space is allowed in <options>\n"
            + "                                 Eg: max_iter=5,epsilon=0.01,positive_weight=2\n"
            + "                      (not used) for run:predict\n";

    if (args.length < 3)
    {
      System.out.println("\n" + cmd);
      System.exit(0);
    }

    // Read the input parameters
    String run = null;
    String ftype = null;
    File dataFile = null;
    File outFile = null;
    double bias = 0;
    File paramFile = null;
    File initFile = null;
    double priorVar = Double.NaN;
    String option = null;
    boolean binaryFeature = false;
    boolean useShort = false;
    boolean computePostVar = true;
    boolean computePostCov = false;

    for (int i = 0; i < args.length; i++)
    {
      if (args[i] == null)
        continue;
      String[] token = args[i].split(":");
      if (token.length < 2)
        cmd_line_error("'" + args[i] + "' is not a valid input parameter string!", cmd);
      for (int k = 2; k < token.length; k++)
        token[1] += ":" + token[k];
      if (token[0].equals("run"))
      {
        run = token[1];
      }
      else if (token[0].equals("ftype"))
      {
        ftype = token[1];
      }
      else if (token[0].equals("data"))
      {
        dataFile = new File(token[1]);
      }
      else if (token[0].equals("out"))
      {
        outFile = new File(token[1]);
      }
      else if (token[0].equals("bias"))
      {
        bias = Double.parseDouble(token[1]);
      }
      else if (token[0].equals("param"))
      {
        paramFile = new File(token[1]);
      }
      else if (token[0].equals("init"))
      {
        initFile = new File(token[1]);
      }
      else if (token[0].equals("priorVar"))
      {
        priorVar = Double.parseDouble(token[1]);
      }
      else if (token[0].equals("option"))
      {
        option = token[1];
      }
      else if (token[0].equals("binaryFeature"))
      {
        binaryFeature = Util.atob(token[1]);
      }
      else if (token[0].equals("useShort"))
      {
        useShort = Util.atob(token[1]);
      }
      else if (token[0].equals("posteriorVar"))
      {
        computePostVar = Util.atob(token[1]);
      }
      else if (token[0].equals("posteriorCov"))
      {
        computePostCov = Util.atob(token[1]);
      }
      else
        cmd_line_error("'" + args[i] + "' is not a valid input parameter string!", cmd);
    }

    if (run == null)
      cmd_line_error("Please specify run:<command>", cmd);
    if (ftype == null)
      cmd_line_error("Please specify ftype:<file_type>", cmd);
    if (dataFile == null)
      cmd_line_error("Please specify data:<file_name>", cmd);
    if (outFile == null)
      cmd_line_error("Please specify out:<file_name>", cmd);

    if (run.equals(RUN_TRAIN))
    {

      Map<String, Double> priorMean = null;
      Map<String, Double> initParam = null;
      if (paramFile != null)
      {
        if (!paramFile.exists())
          cmd_line_error("Param File '" + paramFile.getPath() + "' does not exist", cmd);
        priorMean = Util.readStringDoubleMap(paramFile, "=");
      }
      if(initFile != null)
      {
        if (!initFile.exists())
          cmd_line_error("Init File '" + initFile.getPath() + "' does not exist", cmd);
        initParam = Util.readStringDoubleMap(initFile, "=");
      }

      if (priorVar == Double.NaN)
        cmd_line_error("Please specify priorVar:<var>", cmd);

      if (!dataFile.exists())
        cmd_line_error("Data File '" + dataFile.getPath() + "' does not exist", cmd);

      LibLinearDataset dataset;
      if (binaryFeature)
      {
        dataset = new LibLinearBinaryDataset(bias, useShort);
      }
      else
      {
        dataset = new LibLinearDataset(bias);
      }

      if ("libsvm".equals(ftype))
      {
        dataset.readFromLibSVM(dataFile);
      }
      //else if ("json".equals(ftype))
      //{
      //  dataset.readFromJSON(dataFile);
      //}
      else
        cmd_line_error("Unknown file type 'ftype:" + ftype + "'", cmd);
      
      if(computePostCov == true && computePostVar == false)
        cmd_line_error("Cannot compute posterior covariances with posteriorVar:0", cmd);
      
      LibLinear liblinear = new LibLinear();
      liblinear.setComputeFullPostVar(computePostCov);
      
      liblinear.train(dataset, initParam, priorMean, null, 0.0, priorVar, option, computePostVar);

      PrintStream out = new PrintStream(outFile);
      Util.printStringDoubleMap(out, liblinear.getParamMap(), "=", true);
      out.close();

      if(computePostVar)
      {
        out = new PrintStream(outFile+".var");
        Util.printStringDoubleMap(out, liblinear.getPostVarMap(), "=", true);
        out.close();
        if(computePostCov)
        {
          out = new PrintStream(outFile+".cov");
          Util.printStringListDoubleMap(out, liblinear.getPostVarMatrixMap(), "=");
          out.close();
        }
      }
    }
    else if (run.equals(RUN_PREDICT))
    {

      throw new Exception("run:predict is not supported yet :(");

    }
    else
      cmd_line_error("Unknown run:" + run, cmd);
  }

}
