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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.avro.generic.GenericData;

import com.linkedin.mlease.regression.avro.RegressionPrepareOutput;
import com.linkedin.mlease.regression.avro.feature;
import com.linkedin.mlease.utils.Util;

/**
 * <p>
 * LibLinearDataset contains the training data.
 * </p>
 * 
 * <p>
 * Do not forget to call finish() before using the dataset to train a model.
 * </p>
 * 
 * <p>
 * NOTE:
 * </p>
 * <ul>
 * <li>Feature index starts from 1 (instead of 0)</li>
 * <li>If bias (i.e., intercept) > 0, then the bias value is added to each instance as the
 * last feature (i.e., its feature index = nFeatures)</li>
 * </ul>
 * 
 * @author bchen
 */
public class LibLinearBinaryDataset extends LibLinearDataset
{

  /** array of sparse feature nodes */
  public ArrayList<int[]>   x_int   = null;
  public ArrayList<short[]> x_short = null;
  public boolean            useShort;

  /**
   * Get the number of non-zero features for the ith instance
   * 
   * @param i
   * @return
   */
  public int getNumNonzeroFeatures(int i)
  {
    return (useShort ? x_short.get(i).length : x_int.get(i).length);
  }

  /**
   * Get the jth feature index of instance i
   * 
   * @param i
   * @param j
   * @return
   */
  public int getFeatureIndex(int i, int j)
  {
    return (useShort ? x_short.get(i)[j] : x_int.get(i)[j]);
  }

  /**
   * Construct an empty dataset; If bias > 0 , then the intercept will be added to the end
   * of each feature vector. Set bias = 1 if the input data does not include the intercept
   * feature and you want to have an intercept.
   * 
   * @param bias
   * @param useShort
   *          whether to use short to store feature index
   */
  public LibLinearBinaryDataset(double bias, boolean useShort) throws IOException
  {
    super(bias);
    if (bias != 1 && bias != 0)
    {
      throw new IOException("Bias can only be either 0 or 1: input value = " + bias);
    }
    this.useShort = useShort;
    if (useShort)
    {
      x_short = new ArrayList<short[]>();
    }
    else
    {
      x_int = new ArrayList<int[]>();
    }
  }

  /** Add an instance (a line) in the LibSVM format */
  public void addInstanceLibSVM(String line) throws IOException
  {
    if (finished)
      throw new IOException("Cannot add instances to a finished dataset.");
    StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
    String token;
    try
    {
      token = st.nextToken();
    }
    catch (NoSuchElementException e)
    {
      throw new IOException("Empty line", e);
    }
    try
    {
      y_temp.add(Util.atoi(token));
    }
    catch (NumberFormatException e)
    {
      throw new IOException("Invalid label: " + token, e);
    }
    int m = st.countTokens() / 2;
    int[] x_current;
    if (bias > 0)
    {
      x_current = new int[m + 1];
      x_current[m] = -1;
    }
    else
    {
      x_current = new int[m];
    }
    int indexBefore = 0;
    for (int j = 0; j < m; j++)
    {

      token = st.nextToken();
      int index;
      try
      {
        index = Util.atoi(token);
      }
      catch (NumberFormatException e)
      {
        throw new IOException("Invalid index: " + token, e);
      }

      // assert that indices are valid and sorted
      if (index < 0)
        throw new IOException("Invalid index: " + index);
      if (index <= indexBefore)
        throw new IOException("Indices must be sorted in ascending order");
      indexBefore = index;

      token = st.nextToken();
      try
      {
        double value = Util.atof(token);
        if (value != 1)
          throw new IOException("Cannot handle non-binary features (all feature values have to be 1): "
              + line);
        x_current[j] = index;
      }
      catch (NumberFormatException e)
      {
        throw new IOException("Invalid value: " + token);
      }
    }
    if (m > 0)
    {
      maxFeatureIndex = Math.max(maxFeatureIndex, x_current[m - 1]);
    }
    if (useShort)
    {
      if (maxFeatureIndex >= Short.MAX_VALUE)
        throw new IOException("When using short to store feature indices, you cannot have more than "
            + (Short.MAX_VALUE - 1) + " features!!");
      short[] temp = new short[x_current.length];
      for (int k = 0; k < temp.length; k++)
        temp[k] = (short) x_current[k];
      x_short.add(temp);
    }
    else
    {
      x_int.add(x_current);
    }
  }

  /**
   * Add an instance in the following JSON format: {response=1, features=[{name=F1,
   * term=T1, value=1}, ...], weight=1, offset=0}, where weight and offset are optional.
   * Default: weight = 1 and offset = 0.
   * 
   * @param input
   * @throws IOException
   */
  public void addInstanceJSON(Map<String, ?> input) throws IOException
  {
    if (finished)
      throw new IOException("Cannot add instances to a finished dataset.");
    // response
    int response = Util.getInt(input, "response");
    if (response != 1 && response != 0 && response != -1)
      throw new IOException("response = " + response + " (only 1, 0, -1 are allowed)");
    if (response == 0)
      response = -1;
    y_temp.add(response);
    // weight
    double w = 1;
    if (input.containsKey("weight"))
      w = Util.getDouble(input, "weight");
    if (w != 1)
    {
      if (w < 0)
        throw new IOException("weight = " + w + " (weight cannot < 0)");
      while (weight_temp.size() < y_temp.size() - 1)
        weight_temp.add(1.0);
      weight_temp.add(w);
    }
    // offset
    double o = 0;
    if (input.containsKey("offset"))
      o = Util.getDouble(input, "offset");
    if (o != 0)
    {
      while (offset_temp.size() < y_temp.size() - 1)
        offset_temp.add(0.0);
      offset_temp.add(o);
    }
    // features
    Object temp = input.get("features");
    if (temp == null)
      throw new IOException("features is null");
    if (!(temp instanceof List))
      throw new IOException("features is not a list");
    List<?> features = (List<?>) temp;
    int m = features.size();
    int[] x_current;
    if (bias > 0)
    {
      x_current = new int[m + 1];
      x_current[m] = -1;
    }
    else
    {
      x_current = new int[m];
    }
    for (int i = 0; i < m; i++)
    {
      temp = features.get(i);
      if (!(temp instanceof Map))
        throw new IOException("features[" + i + "] is not a map");
      Map<String, ?> feature = (Map<String, ?>) temp;
      String name = Util.getString(feature, "name", false);
      String term = Util.getString(feature, "term", true);
      if (!"".equals(term))
        name = name + "\u0001" + term;
      if (feature.containsKey("value"))
      {
        double value = Util.getDouble(feature, "value");
        if (value != 1)
          throw new IOException("Cannot handle non-binary feature value (all feature values have to be 1; or just do not specify the value): "
              + feature.toString());
      }
      if (featureIndex == null)
        featureIndex = new HashMap<String, Integer>();
      if (featureName == null)
        featureName = new ArrayList<String>();
      if (featureIndex.size() != featureName.size())
        throw new IOException("featureIndex.size() != featureName.size()");
      Integer index = featureIndex.get(name);
      if (index == null)
      {
        if (INTERCEPT_NAME.equals(name))
          throw new IOException("feature name cannot be " + INTERCEPT_NAME);
        maxFeatureIndex++;
        featureIndex.put(name, maxFeatureIndex);
        featureName.add(name);
        if (featureName.size() != maxFeatureIndex)
          throw new IOException("featureName.size() != maxFeatureIndex");
        index = maxFeatureIndex;
      }
      x_current[i] = index;
    }
    if (m > 1)
      Arrays.sort(x_current, 0, m);

    if (useShort)
    {
      if (maxFeatureIndex >= Short.MAX_VALUE)
        throw new IOException("When using short to store feature indices, you cannot have more than "
            + (Short.MAX_VALUE - 1) + " features!!");
      short[] temp2 = new short[x_current.length];
      for (int k = 0; k < temp2.length; k++)
        temp2[k] = (short) x_current[k];
      x_short.add(temp2);
    }
    else
    {
      x_int.add(x_current);
    }
  }

  public void addInstanceAvro(GenericData.Record input) throws IOException
  {
    if (finished)
      throw new IOException("Cannot add instances to a finished dataset.");
    // response
    int response = Util.getIntAvro(input, "response");
    if (response != 1 && response != 0 && response != -1)
      throw new IOException("response = " + response + " (only 1, 0, -1 are allowed)");
    if (response == 0)
      response = -1;
    y_temp.add(response);
    // weight
    double w = 1;
    if (input.get("weight") != null)
      w = Util.getDoubleAvro(input, "weight");
    if (w != 1)
    {
      if (w < 0)
        throw new IOException("weight = " + w + " (weight cannot < 0)");
      while (weight_temp.size() < y_temp.size() - 1)
        weight_temp.add(1.0);
      weight_temp.add(w);
    }
    // offset
    double o = 0;
    if (input.get("offset") != null)
      o = Util.getDoubleAvro(input, "offset");
    if (o != 0)
    {
      while (offset_temp.size() < y_temp.size() - 1)
        offset_temp.add(0.0);
      offset_temp.add(o);
    }
    // features
    Object temp = input.get("features");
    if (temp == null)
      throw new IOException("features is null");
    if (!(temp instanceof List))
      throw new IOException("features is not a list");
    List<?> features = (List<?>) temp;
    int m = features.size();
    int[] x_current;
    if (bias > 0)
    {
      x_current = new int[m + 1];
      x_current[m] = -1;
    }
    else
    {
      x_current = new int[m];
    }
    for (int i = 0; i < m; i++)
    {
      temp = features.get(i);
      if (!(temp instanceof GenericData.Record))
        throw new IOException("features[" + i + "] is not a record");
      GenericData.Record feature = (GenericData.Record) temp;
      String name = Util.getStringAvro(feature, "name", false);
      String term = Util.getStringAvro(feature, "term", true);
      if (!"".equals(term))
        name = name + "\u0001" + term;
      if (feature.get("value") != null)
      {
        double value = Util.getDoubleAvro(feature, "value");
        if (value != 1)
          throw new IOException("Cannot handle non-binary feature value (all feature values have to be 1; or just do not specify the value): "
              + feature.toString());
      }
      if (featureIndex == null)
        featureIndex = new HashMap<String, Integer>();
      if (featureName == null)
        featureName = new ArrayList<String>();
      if (featureIndex.size() != featureName.size())
        throw new IOException("featureIndex.size() != featureName.size()");
      Integer index = featureIndex.get(name);
      if (index == null)
      {
        if (INTERCEPT_NAME.equals(name))
          throw new IOException("feature name cannot be " + INTERCEPT_NAME);
        maxFeatureIndex++;
        featureIndex.put(name, maxFeatureIndex);
        featureName.add(name);
        if (featureName.size() != maxFeatureIndex)
          throw new IOException("featureName.size() != maxFeatureIndex");
        index = maxFeatureIndex;
      }
      x_current[i] = index;
    }
    if (m > 1)
      Arrays.sort(x_current, 0, m);

    if (useShort)
    {
      if (maxFeatureIndex >= Short.MAX_VALUE)
        throw new IOException("When using short to store feature indices, you cannot have more than "
            + (Short.MAX_VALUE - 1) + " features!!");
      short[] temp2 = new short[x_current.length];
      for (int k = 0; k < temp2.length; k++)
        temp2[k] = (short) x_current[k];
      x_short.add(temp2);
    }
    else
    {
      x_int.add(x_current);
    }
  }

  public void addInstanceAvro(RegressionPrepareOutput input) throws IOException
  {
    if (finished)
      throw new IOException("Cannot add instances to a finished dataset.");
    // response
    int response = input.response;
    if (response != 1 && response != 0 && response != -1)
      throw new IOException("response = " + response + " (only 1, 0, -1 are allowed)");
    if (response == 0)
      response = -1;
    y_temp.add(response);
    // weight
    double w = input.weight;
    if (w != 1)
    {
      if (w < 0)
        throw new IOException("weight = " + w + " (weight cannot < 0)");
      while (weight_temp.size() < y_temp.size() - 1)
        weight_temp.add(1.0);
      weight_temp.add(w);
    }
    // offset
    double o = input.offset;
    if (o != 0)
    {
      while (offset_temp.size() < y_temp.size() - 1)
        offset_temp.add(0.0);
      offset_temp.add(o);
    }
    // features
    List<feature> features = input.features;
    int m = features.size();
    int[] x_current;
    if (bias > 0)
    {
      x_current = new int[m + 1];
      x_current[m] = -1;
    }
    else
    {
      x_current = new int[m];
    }
    for (int i = 0; i < m; i++)
    {
      String name = features.get(i).name.toString();
      String term = features.get(i).term.toString();
      if (!"".equals(term))
        name = name + "\u0001" + term;
      double value = features.get(i).value;
      if (value != 1)
        throw new IOException("Cannot handle non-binary feature value (all feature values have to be 1; or just do not specify the value): "
            + features.get(i).toString());
      if (featureIndex == null)
        featureIndex = new HashMap<String, Integer>();
      if (featureName == null)
        featureName = new ArrayList<String>();
      if (featureIndex.size() != featureName.size())
        throw new IOException("featureIndex.size() != featureName.size()");
      Integer index = featureIndex.get(name);
      if (index == null)
      {
        if (INTERCEPT_NAME.equals(name))
          throw new IOException("feature name cannot be " + INTERCEPT_NAME);
        maxFeatureIndex++;
        featureIndex.put(name, maxFeatureIndex);
        featureName.add(name);
        if (featureName.size() != maxFeatureIndex)
          throw new IOException("featureName.size() != maxFeatureIndex");
        index = maxFeatureIndex;
      }
      x_current[i] = index;
    }
    if (m > 1)
      Arrays.sort(x_current, 0, m);

    if (useShort)
    {
      if (maxFeatureIndex >= Short.MAX_VALUE)
        throw new IOException("When using short to store feature indices, you cannot have more than "
            + (Short.MAX_VALUE - 1) + " features!!");
      short[] temp2 = new short[x_current.length];
      for (int k = 0; k < temp2.length; k++)
        temp2[k] = (short) x_current[k];
      x_short.add(temp2);
    }
    else
    {
      x_int.add(x_current);
    }
  }

  /**
   * Reset the dataset by ignoring the instances that have been added to it.
   */
  public void reset() throws IOException
  {
    if (finished)
      throw new IOException("Cannot reset a finished dataset.");
    n = 0;
    l = 0;
    y_temp.clear();
    if (x_int != null)
      x_int.clear();
    if (x_short != null)
      x_short.clear();
    offset_temp.clear();
    weight_temp.clear();
    maxFeatureIndex = 0;
  }

  /**
   * Call this method before training a model
   * 
   * @throws IOException
   */
  public void finish() throws IOException
  {
    if (finished)
      throw new IOException("Cannot finish a finished dataset.");
    l = y_temp.size();
    n = maxFeatureIndex;
    if (bias > 0)
    {
      n++;
      if (n == 1 && (featureIndex == null || featureName == null))
      {
        featureIndex = new HashMap<String, Integer>();
        featureName = new ArrayList<String>();
      }
      if (featureIndex != null)
        featureIndex.put(INTERCEPT_NAME, n);
      if (featureName != null)
        featureName.add(INTERCEPT_NAME);
    }

    if (useShort)
    {
      for (int i = 0; i < l; i++)
      {
        short[] x_current = x_short.get(i);
        if (bias > 0)
        {
          assert x_current[x_current.length - 1] == -1;
          x_current[x_current.length - 1] = (short) (maxFeatureIndex + 1);
        }
      }
    }
    else
    {
      for (int i = 0; i < l; i++)
      {
        int[] x_current = x_int.get(i);
        if (bias > 0)
        {
          assert x_current[x_current.length - 1] == -1;
          x_current[x_current.length - 1] = maxFeatureIndex + 1;
        }
      }
    }

    y = new int[l];
    for (int i = 0; i < l; i++)
      y[i] = y_temp.get(i);

    offset = new double[l];
    if (offset_temp.size() > 0)
    {
      int i;
      for (i = 0; i < offset_temp.size(); i++)
        offset[i] = offset_temp.get(i);
      for (; i < l; i++)
        offset[i] = 0;
    }
    else
    {
      for (int i = 0; i < offset.length; i++)
        offset[i] = 0;
    }

    weight = new double[l];
    if (weight_temp.size() > 0)
    {
      int i;
      for (i = 0; i < weight_temp.size(); i++)
        weight[i] = weight_temp.get(i);
      for (; i < l; i++)
        weight[i] = 1;
    }
    else
    {
      for (int i = 0; i < weight.length; i++)
        weight[i] = 1;
    }

    sanity_check(1);
    y_temp.clear();
    offset_temp.clear();
    weight_temp.clear();
    finished = true;
  }

  /**
   * Each line is in the following form: response TAB features TAB weight TAB offset where
   * features is in the following form: space-separated list of NAME=VALUE e.g.,
   * "feature1=3  feature3=1" or "1=3  3=1"
   */
  public String toString()
  {
    if (l == 0)
      return "";
    StringBuilder out = new StringBuilder();
    try
    {
      sanity_check(1);
    }
    catch (IOException e)
    {
      return "ERROR: " + e.getMessage() + "\n";
    }
    for (int i = 0; i < l; i++)
    {
      out.append(y[i]);
      out.append("\t");

      if (useShort)
      {
        for (int j = 0; j < x_short.get(i).length; j++)
        {
          if (j > 0)
            out.append(" ");
          String name;
          if (featureName == null)
          {
            name = x_short.get(i)[j] + "";
          }
          else
          {
            name = featureName.get(x_short.get(i)[j] - 1);
          }
          out.append(name + "=1");
        }
      }
      else
      {
        for (int j = 0; j < x_int.get(i).length; j++)
        {
          if (j > 0)
            out.append(" ");
          String name;
          if (featureName == null)
          {
            name = x_int.get(i)[j] + "";
          }
          else
          {
            name = featureName.get(x_int.get(i)[j] - 1);
          }
          out.append(name + "=1");
        }
      }

      if (weight != null)
      {
        out.append("\t" + weight[i]);
      }
      else if (offset != null)
      {
        out.append("\t1");
      }
      if (offset != null)
      {
        out.append("\t" + offset[i]);
      }
      out.append("\n");
    }
    return out.toString();
  }

  /**
   * Sanity check
   * 
   * @param level
   *          0: check dimensions. 1: check feature name <-> index mapping. 2: check each
   *          data record.
   * @throws IOException
   */
  public void sanity_check(int level) throws IOException
  {
    if (l != y.length)
      throw new IOException("l = " + l + ", but y.length = " + y.length);
    if ((!useShort) && l != x_int.size())
      throw new IOException("l = " + l + ", but x.size() = " + x_int.size());
    if (useShort && l != x_short.size())
      throw new IOException("l = " + l + ", but x.size() = " + x_short.size());
    if (offset != null && l != offset.length)
      throw new IOException("l = " + l + ", but offset.length =" + offset.length);
    if (weight != null && l != weight.length)
      throw new IOException("l = " + l + ", but weight.length =" + weight.length);
    if (featureIndex != null && featureName == null)
      throw new IOException("featureIndex != null && featureName == null");
    if (featureIndex == null && featureName != null)
      throw new IOException("featureIndex == null && featureName != null");
    if (featureIndex != null)
    {
      if (bias > 0 && (!INTERCEPT_NAME.equals(featureName.get(featureName.size() - 1))))
        throw new IOException("The last feature is not " + INTERCEPT_NAME);
      if (featureIndex.size() != featureName.size())
        throw new IOException("featureIndex.size()=" + featureIndex.size()
            + ", but featureName.size()=" + featureName.size());
      if (featureIndex.size() != n)
        throw new IOException("featureIndex.size()=" + featureIndex.size() + ", but n="
            + n);
    }
    if (level >= 1)
    {
      if (featureIndex != null)
      {
        Set<Map.Entry<String, Integer>> entrySet = featureIndex.entrySet();
        Iterator<Map.Entry<String, Integer>> iter = entrySet.iterator();
        while (iter.hasNext())
        {
          Map.Entry<String, Integer> entry = iter.next();
          if (featureName.get(entry.getValue() - 1) != entry.getKey())
            throw new IOException("featureName[" + (entry.getValue() - 1) + "] = "
                + featureName.get(entry.getValue() - 1) + ", instead of "
                + entry.getKey());
        }
      }
    }
    if (level >= 2)
    {
      if (useShort)
      {
        for (int i = 0; i < x_short.size(); i++)
        {
          for (int j = 0; j < x_short.get(i).length; j++)
          {
            int index = x_short.get(i)[j];
            if (index < 1 || index > n)
              throw new IOException("feature index out of bound x[" + i + "][" + j
                  + "].index=" + index);
          }
        }
      }
      else
      {
        for (int i = 0; i < x_int.size(); i++)
        {
          for (int j = 0; j < x_int.get(i).length; j++)
          {
            int index = x_int.get(i)[j];
            if (index < 1 || index > n)
              throw new IOException("feature index out of bound x[" + i + "][" + j
                  + "].index=" + index);
          }
        }
      }
      for (int i = 0; i < y.length; i++)
      {
        if (y[i] != 1 && y[i] != -1 && y[i] != 0)
          throw new IOException("y[" + i + "] = " + y[i]);
      }
    }
  }

  /**
   * For testing purposes only
   * 
   * @param args
   */
  public static void main(String[] args) throws Exception
  {
    if (args.length != 2)
    {
      System.out.println("Please give two input parameters: <type> <filename>, where type = libsvm or json");
      System.exit(0);
    }
    String type = args[0];
    File file = new File(args[1]);
    if (!file.exists())
    {
      System.out.println("File " + file.toString() + " does not exist");
      System.exit(0);
    }
    LibLinearBinaryDataset d = new LibLinearBinaryDataset(0, false);
    if ("libsvm".equals(type))
    {
      d.readFromLibSVM(file);
    }
    //else if ("json".equals(type))
    //{
    //  d.readFromJSON(file);
    //}
    else
    {
      System.out.println("Unknown file type: " + type);
      System.exit(0);
    }
    d.sanity_check(10);
    System.out.println(d.toString());
  }

}
