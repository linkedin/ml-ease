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

import org.apache.hadoop.mapred.Reporter;

import com.linkedin.mlease.regression.liblinearfunc.LibLinearDataset;

import de.bwaldvogel.liblinear.Feature;

;

/**
 * <p>
 * Logistic Regression with a Gaussian prior
 * </p>
 * 
 * <pre>
 * Let score_i(w) = w'x[i,] + data.offset[i].
 *      prob_i(w) = (1 + exp(-y[i] * score_i(w)))^-1
 *      
 * loss(w) = 
 *      (1/2) * sum_k { (w[k] - priorMean[k])^2 / priorVar[k] }
 *    + sum_i { data.weight[i] * log(1 + exp(-y[i] * score_i(w))) }
 *    
 * loss'(w) =
 *      sum_k { (w[k] - priorMean[k]) / priorVar[k] }
 *    + sum_i { data.weight[i] * (prob_i(w) - 1) y[i] x[i,] }
 *    
 * loss''(w) = diag(1/priorVar) + X' D X,
 * 
 *    where D[i,i] = data.weight[i] * prob_i(w) * (prob_i(w) - 1)
 * </pre>
 * 
 * @author bchen
 */
public class LogisticRegressionL2 implements LibLinearFunction
{

  protected final double[]         weight;
  protected final double[]         z;
  protected final double[]         D;
  protected final LibLinearDataset data;
  protected final double[]         priorMean;
  protected final double[]         priorVar_inv;
  protected final double           multiplier;

  Reporter                         reporter        = null;
  long                             reportFrequency = 10000000;

  public void setReporter(Reporter rep, long n)
  {
    reporter = rep;
    reportFrequency = n;
  }

  long counter = 0;

  public LogisticRegressionL2(LibLinearDataset dataset,
                              double[] priorMean,
                              double[] priorVar,
                              double multiplier,
                              double Cp,
                              double Cn)
  {

    if (reporter != null)
      reporter.progress();

    int i;
    int l = dataset.l;
    int[] y = dataset.y;

    data = dataset;

    z = new double[l];
    D = new double[l];
    weight = new double[l];

    for (i = 0; i < l; i++)
    {
      if (y[i] == 1)
        weight[i] = Cp * data.weight[i];
      else
        weight[i] = Cn * data.weight[i];
    }

    if (reporter != null)
      reporter.progress();

    this.priorMean = priorMean;
    this.multiplier = multiplier;

    priorVar_inv = new double[data.nFeatures()];
    for (i = 0; i < priorVar_inv.length; i++)
      priorVar_inv[i] = 1.0 / priorVar[i];

    if (reporter != null)
      reporter.progress();
  }

  protected void Xv(double[] v, double[] Xv)
  {

    for (int i = 0; i < data.l; i++)
    {
      Xv[i] = 0;
      for (Feature s : data.x[i])
      {
        Xv[i] += v[s.getIndex() - 1] * s.getValue();
      }
      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
  }

  protected void XTv(double[] v, double[] XTv)
  {
    int l = data.l;
    int w_size = data.nFeatures();
    Feature[][] x = data.x;

    for (int i = 0; i < w_size; i++)
      XTv[i] = 0;

    for (int i = 0; i < l; i++)
    {
      for (Feature s : x[i])
      {
        XTv[s.getIndex() - 1] += v[i] * s.getValue();
      }
      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
  }

  /**
   * loss(w) = (1/2) * sum_k { (w[k] - priorMean[k])^2 / priorVar[k] } + sum_i {
   * data.weight[i] * log(1 + exp(-y[i] * score_i(w))) }
   */
  public double fun(double[] w)
  {
    int i;
    double f = 0;
    int[] y = data.y;
    int l = data.l;
    int w_size = data.nFeatures();

    if (reporter != null)
      reporter.progress();

    Xv(w, z);

    for (i = 0; i < l; i++)
    {

      z[i] += data.offset[i];
      // z[i] = score_i(w) = w'x[i,] + data.offset[i]
      double yz = y[i] * z[i];
      if (yz >= 0)
        f += weight[i] * Math.log1p(Math.exp(-yz));
      else
        f += weight[i] * (-yz + Math.log1p(Math.exp(yz)));

      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
    f = 2.0 * f;
    for (i = 0; i < w_size; i++)
    {
      double temp = w[i] - priorMean[i];
      f += temp * temp * priorVar_inv[i];
    }
    f /= 2.0;

    return (multiplier * f);
  }

  /**
   * loss'(w) = sum_k { (w[k] - priorMean[k]) / priorVar[k] } + sum_i { data.weight[i] *
   * (prob_i(w) - 1) y[i] x[i,] }
   */
  public void grad(double[] w, double[] g)
  {
    int i;
    int[] y = data.y;
    int l = data.l;
    int w_size = data.nFeatures();

    if (reporter != null)
      reporter.progress();

    for (i = 0; i < l; i++)
    {
      z[i] = 1 / (1 + Math.exp(-y[i] * z[i]));
      // z[i] = prob_i(w) = (1 + exp(-y[i] * score_i(w)))^-1
      D[i] = z[i] * (1 - z[i]);
      // D[i] = prob_i(w) * (1 - prob_i(w))
      z[i] = weight[i] * (z[i] - 1) * y[i];

      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
    XTv(z, g);

    for (i = 0; i < w_size; i++)
      g[i] = ((w[i] - priorMean[i]) * priorVar_inv[i] + g[i]) * multiplier;
  }

  /**
   * loss''(w) = diag(1/priorVar) + X' D X, where D[i,i] = data.weight[i] * prob_i(w) *
   * (1 - prob_i(w))
   */
  public void Hv(double[] s, double[] Hs)
  {
    int i;
    int l = data.l;
    int w_size = data.nFeatures();
    double[] wa = new double[l];

    if (reporter != null)
      reporter.progress();

    Xv(s, wa);
    for (i = 0; i < l; i++)
      wa[i] = weight[i] * D[i] * wa[i];

    XTv(wa, Hs);
    for (i = 0; i < w_size; i++)
      Hs[i] = (s[i] * priorVar_inv[i] + Hs[i]) * multiplier;
  }

  
  /**
   * loss''(w) = diag(1/priorVar) + X' D X,
   * where D[i,i] = data.weight[i] * prob_i(w) * (1 - prob_i(w))
   * 
   * H[m][n] = (m == n ? 1/priorVar[m] : 0) +
   *           sum_i D[i,i] * X[i,m] * X[i,n]
   */
  public void hessian(double[] w, double[][] H){
    for(int k=0; k<data.n; k++) H[k][k] = priorVar_inv[k];
    for(int i=0; i<data.l; i++){
      double score = 0;
      for (Feature s : data.x[i])
      {
        score += w[s.getIndex() - 1] * s.getValue();
      }
      score += data.offset[i];
      // score = w'x[i,] + data.offset[i]
      double p = 1.0 / (1.0 + Math.exp(-data.y[i] * score));
      double D_ii = weight[i] * p * (1-p);
      
      // Fill in H[m][n] for m >= n
      int prev_index = Integer.MIN_VALUE;
      for (Feature s : data.x[i])
      {
        int m = s.getIndex() - 1;
        // check whether features are sorted by index
        if(m <= prev_index) throw new RuntimeException("The input features are not sorted by feature index values");
        prev_index = m;
        
        for (Feature t : data.x[i])
        {
          int n = t.getIndex() - 1;
          H[m][n] += D_ii * s.getValue() * t.getValue();
          if(m == n) break;
        }
      }

      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
    // Fill in H[m][n] for m < n
    for(int m=0; m<H.length; m++)
    {
      for(int n=m+1; n<H.length; n++) H[m][n] = H[n][m];
    }
  }

  /**
   * H[k] = 1/priorVar[k] + sum_i data.weight[i] * 
   *                              prob_i(w) * (1 - prob_i(w)) *
   *                              X[i,k] * X[i,k]
   */
  public void hessianDiagonal(double[] w, double[] H){
    for(int k=0; k<data.n; k++) H[k] = priorVar_inv[k];
    for(int i=0; i<data.l; i++){
      double score = 0;
      for (Feature s : data.x[i])
      {
        score += w[s.getIndex() - 1] * s.getValue();
      }
      score += data.offset[i];
      // score = w'x[i,] + data.offset[i]
      double p = 1.0 / (1.0 + Math.exp(-data.y[i] * score));
      double q = weight[i] * p * (1-p);
      
      for (Feature s : data.x[i])
      {
        int k = s.getIndex() - 1;
        H[k] += q * s.getValue() * s.getValue();
      }

      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
  }

  public int get_nr_variable()
  {
    return data.nFeatures();
  }

}
