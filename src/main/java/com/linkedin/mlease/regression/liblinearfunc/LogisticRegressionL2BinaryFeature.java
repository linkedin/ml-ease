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

import com.linkedin.mlease.regression.liblinearfunc.LibLinearBinaryDataset;

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
public class LogisticRegressionL2BinaryFeature extends LogisticRegressionL2
{

  public LogisticRegressionL2BinaryFeature(LibLinearBinaryDataset dataset,
                                           double[] priorMean,
                                           double[] priorVar,
                                           double multiplier,
                                           double Cp,
                                           double Cn)
  {
    super(dataset, priorMean, priorVar, multiplier, Cp, Cn);
  }

  @Override
  protected void Xv(double[] v, double[] Xv)
  {
    LibLinearBinaryDataset d = (LibLinearBinaryDataset) data;
    for (int i = 0; i < data.l; i++)
    {
      Xv[i] = 0;
      for (int j = 0; j < d.getNumNonzeroFeatures(i); j++)
      {
        Xv[i] += v[d.getFeatureIndex(i, j) - 1];
      }
      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
  }

  @Override
  protected void XTv(double[] v, double[] XTv)
  {
    int l = data.l;
    int w_size = data.nFeatures();
    LibLinearBinaryDataset d = (LibLinearBinaryDataset) data;
    for (int i = 0; i < w_size; i++)
      XTv[i] = 0;

    for (int i = 0; i < l; i++)
    {
      for (int j = 0; j < d.getNumNonzeroFeatures(i); j++)
      {
        XTv[d.getFeatureIndex(i, j) - 1] += v[i];
      }
      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
  }

  /**
   * H[k] = 1/priorVar[k] + sum_i data.weight[i] * 
   *                              prob_i(w) * (1 - prob_i(w)) *
   *                              X[i,k] * X[i,k]
   */
  @Override
  public void hessianDiagonal(double[] w, double[] H){
    LibLinearBinaryDataset d = (LibLinearBinaryDataset) data;
    for(int k=0; k<data.n; k++) H[k] = priorVar_inv[k];
    for(int i=0; i<data.l; i++){
      double score = 0;
      for (int j = 0; j < d.getNumNonzeroFeatures(i); j++)
      {
        score += w[d.getFeatureIndex(i, j) - 1];
      }
      score += data.offset[i];
      // score = w'x[i,] + data.offset[i]
      double p = 1.0 / (1.0 + Math.exp(-data.y[i] * score));
      double q = weight[i] * p * (1-p);
      
      for (int j = 0; j < d.getNumNonzeroFeatures(i); j++)
      {
        int k = d.getFeatureIndex(i, j) - 1;
        H[k] += q;
      }

      counter = (counter + 1) % reportFrequency;
      if (counter == 0 && reporter != null)
        reporter.progress();
    }
  }

  /**
   * loss''(w) = diag(1/priorVar) + X' D X,
   * where D[i,i] = data.weight[i] * prob_i(w) * (1 - prob_i(w))
   * 
   * H[m][n] = (m == n ? 1/priorVar[m] : 0) +
   *           sum_i D[i,i] * X[i,m] * X[i,n]
   */
  public void hessian(double[] w, double[][] H){
    LibLinearBinaryDataset d = (LibLinearBinaryDataset) data;
    for(int k=0; k<data.n; k++) H[k][k] = priorVar_inv[k];
    for(int i=0; i<data.l; i++){
      double score = 0;
      for (int j = 0; j < d.getNumNonzeroFeatures(i); j++)
      {
        score += w[d.getFeatureIndex(i, j) - 1];
      }
      score += data.offset[i];
      // score = w'x[i,] + data.offset[i]
      double p = 1.0 / (1.0 + Math.exp(-data.y[i] * score));
      double D_ii = weight[i] * p * (1-p);
      
      // Fill in H[m][n] for m >= n
      int prev_index = Integer.MIN_VALUE;
      for (int j = 0; j < d.getNumNonzeroFeatures(i); j++)
      {
        int m = d.getFeatureIndex(i, j) - 1;
        // check whether features are sorted by index
        if(m <= prev_index) throw new RuntimeException("The input features are not sorted by feature index values");
        prev_index = m;

        for (int k = 0; k < d.getNumNonzeroFeatures(i); k++)
        {
          int n = d.getFeatureIndex(i, k) - 1;
          H[m][n] += D_ii;
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

}
