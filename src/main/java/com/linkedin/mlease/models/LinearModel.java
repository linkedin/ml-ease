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

package com.linkedin.mlease.models;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.apache.avro.generic.GenericData;

import com.linkedin.mlease.utils.Util;
import com.linkedin.mlease.avro.feature;

/**
 * LinearModel is the class for all the operations on the linear model coefficient level
 * 
 * @author lizhang
 * 
 */
public class LinearModel
{

  private double              _intercept;
  // The coefficients excluding the intercept
  private Map<String, Double> _coefficients;

  public LinearModel()
  {
    _intercept = 0;
    _coefficients = new HashMap<String, Double>();
  }

  public LinearModel(double intercept, Map<String, Double> coefficients)
  {
    _intercept = intercept;
    _coefficients = new HashMap<String, Double>();
    _coefficients.putAll(coefficients);
  }

  /**
   * intercept_key gives the key of the intercept in the hashmap such as "(INTERCEPT)".
   * 
   * @param intercept_key
   * @param coefficients
   * @throws IOException
   */
  public LinearModel(String intercept_key, Map<String, Double> coefficients) throws IOException
  {
    if (!coefficients.containsKey(intercept_key))
    {
      throw new IOException("intercept_key does not exist in the hashmap coefficients!");
    }
    _intercept = coefficients.get(intercept_key);
    _coefficients = new HashMap<String, Double>();
    _coefficients.putAll(coefficients);
    _coefficients.remove(intercept_key);
  }

  /**
   * intercept_key gives the key of the intercept, such as "0". modelstr is the model
   * string with format such as "0=1.0 5=2.0 10=1.0" or "abc=1.0 def=2.0". innerdelim is
   * the inner delimiter such as "=" outerdelim is the outer delimiter such as " "
   * 
   * @param intercept_key
   * @param modelstr
   * @throws IOException
   */
  public LinearModel(String intercept_key,
                     String modelstr,
                     String innerdelim,
                     String outerdelim) throws IOException
  {
    _intercept = 0;
    _coefficients = new HashMap<String, Double>();
    String str = modelstr.replaceAll("(\\r|\\n)", "");
    String[] token = str.split(outerdelim);
    for (int i = 0; i < token.length; i++)
    {
      String[] s = token[i].split(innerdelim);
      if (s.length != 2)
        throw new IOException("Model format is wrong! " + modelstr);
      if (!s[0].equals(intercept_key))
      {
        _coefficients.put(s[0], Util.atof(s[1]));
      }
      else
      {
        _intercept = Util.atof(s[1]);
      }
    }
  }

  public LinearModel(String intercept_key, List<?> modellist) throws IOException
  {
    _intercept = 0;
    _coefficients = new HashMap<String, Double>();
    for (int i = 0; i < modellist.size(); i++)
    {
      Object temp = modellist.get(i);
      if (!(temp instanceof Map) && !(temp instanceof GenericData.Record))
        throw new IOException("features[" + i + "] is not a map or avro record");
      if (temp instanceof Map)
      {
        Map<String, ?> feature = (Map<String, ?>) temp;
        String name = Util.getString(feature, "name", false);
        String term = Util.getString(feature, "term", true);
        if (!"".equals(term))
          name = name + "\u0001" + term;
        double value = Util.getDouble(feature, "value");
        if (name.equals(intercept_key))
        {
          _intercept = value;
        }
        else
        {
          _coefficients.put(name, value);
        }
      }
      else
      {
        GenericData.Record feature = (GenericData.Record) temp;
        String name = Util.getStringAvro(feature, "name", false);
        String term = Util.getStringAvro(feature, "term", true);
        if (!"".equals(term))
          name = name + "\u0001" + term;
        double value = Util.getDoubleAvro(feature, "value");
        if (name.equals(intercept_key))
        {
          _intercept = value;
        }
        else
        {
          _coefficients.put(name, value);
        }
      }
    }
  }

  /**
   * x = ax
   * 
   * @param a
   */
  public void rescale(double a)
  {
    _intercept = a * _intercept;
    for (String k : _coefficients.keySet())
    {
      _coefficients.put(k, _coefficients.get(k) * a);
    }
  }

  /**
   * Linear combination of this model with another model y. x = ax + by, where x and y are
   * both LinearModel.
   * 
   * @param a
   * @param b
   * @param y
   * @return
   */
  public void linearCombine(double a, double b, LinearModel y)
  {
    _intercept = a * _intercept + b * y.getIntercept();
    HashSet<String> keys = new HashSet<String>(_coefficients.keySet());
    keys.addAll(y.getCoefficients().keySet());
    Iterator<String> iter = keys.iterator();
    while (iter.hasNext())
    {
      String key = iter.next();
      double value = 0;
      if (_coefficients.containsKey(key))
      {
        value = value + a * _coefficients.get(key);
      }
      if (y.getCoefficients().containsKey(key))
      {
        value = value + b * y.getCoefficients().get(key);
      }
      _coefficients.put(key, value);
    }
  }

  public void linearCombine(double a, double b, LinearModel y, Map<String, Double> bmap)
  {
    _intercept = a * _intercept + b * y.getIntercept();
    HashSet<String> keys = new HashSet<String>(_coefficients.keySet());
    keys.addAll(y.getCoefficients().keySet());
    Iterator<String> iter = keys.iterator();
    while (iter.hasNext())
    {
      String key = iter.next();
      double value = 0;
      if (_coefficients.containsKey(key))
      {
        value = value + a * _coefficients.get(key);
      }
      if (y.getCoefficients().containsKey(key))
      {
        if (!bmap.containsKey(key))
        {
          value = value + b * y.getCoefficients().get(key);
        }
        else
        {
          value = value + bmap.get(key) * y.getCoefficients().get(key);
        }
      }
      _coefficients.put(key, value);
    }
  }

  /**
   * Get the x'beta
   * 
   * @param keys
   * @param values
   * @param num_click_replicates
   * @return
   * @throws IOException
   */
  private double eval(String[] keys, double[] values, int num_click_replicates) throws IOException
  {
    double result =
        -Math.log(num_click_replicates - 1 + num_click_replicates * Math.exp(-_intercept));
    if (keys.length != values.length)
    {
      throw new IOException("The length of keys and values must be equal!");
    }
    for (int i = 0; i < keys.length; i++)
    {
      if (_coefficients.containsKey(keys[i]))
      {
        result += _coefficients.get(keys[i]) * values[i];
      }
    }
    return result;
  }

  private double eval(String[] keys, double[] values) throws IOException
  {
    return eval(keys, values, 1);
  }

  /**
   * If loglik=F, it returns the evaluation of the x'beta given a line in libsvm format.
   * If loglik=T, it returns the test log-liklihood.
   * 
   * @param line
   * @return
   * @throws IOException
   */
  public double evalInstanceLibSVM(String line, boolean loglik) throws IOException
  {
    StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
    String token;
    int y = 0;
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
      y = Util.atoi(token);
    }
    catch (NumberFormatException e)
    {
      throw new IOException("Invalid label: " + token, e);
    }
    if (y != 1 && y != 0 && y != -1)
      throw new IOException("response = " + y);
    int m = st.countTokens() / 2;
    String[] keys = new String[m];
    double[] values = new double[m];
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
      keys[j] = String.valueOf(index);
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
        values[j] = value;
      }
      catch (NumberFormatException e)
      {
        throw new IOException("Invalid value: " + token);
      }
    }
    if (!loglik)
    {
      return eval(keys, values);
    }
    else
    {
      double xbeta = eval(keys, values);
      if (y == 1)
      {
        return -Math.log(1 + Math.exp(-xbeta));
      }
      else
      {
        return -Math.log(1 + Math.exp(xbeta));
      }
    }
  }

  /**
   * 
   * @param input
   * @param schema_type
   * @param loglik
   * @param num_click_replicates
   * @param ignore_value
   * @return
   * @throws IOException
   */
  public double evalInstanceJSON(Map<String, ?> input,
                                 int schema_type,
                                 boolean loglik,
                                 int num_click_replicates,
                                 boolean ignore_value) throws IOException
  {
    // response
    int y = Util.getResponse(input);
    if (y != 1 && y != 0 && y != -1)
      throw new IOException("response = " + y);
    // offset
    double o = 0;
    if (input.containsKey("offset"))
      o = Util.getDouble(input, "offset");
    // features
    String[] keys = null;
    double[] values = null;
    if (schema_type == 1)
    {
      Object temp = input.get("features");
      if (temp == null)
        throw new IOException("features is null");
      if (!(temp instanceof List))
        throw new IOException("features is not a list");
      List<?> features = (List<?>) temp;
      int m = features.size();
      keys = new String[m];
      values = new double[m];
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
        double value = 1.0;
        if (!ignore_value)
          value = Util.getDouble(feature, "value");
        keys[i] = name;
        values[i] = value;
      }
    }
    if (schema_type == 2)
    {
      if (!input.containsKey("indexes") || !input.containsKey("values"))
      {
        throw new IOException("Data should contain indexes and values columns!");
      }
      Object tmp1 = input.get("indexes");
      Object tmp2 = input.get("values");
      if (!(tmp1 instanceof List))
        throw new IOException("indexes is not a list!");
      if (!(tmp2 instanceof List))
        throw new IOException("values is not a list!");
      List<?> indexes = (List<?>) tmp1;
      List<?> Values = (List<?>) tmp2;
      if (indexes.size() != Values.size())
        throw new IOException("Size of indexes and values do not match!");
      keys = new String[indexes.size()];
      values = new double[indexes.size()];
      for (int i = 0; i < indexes.size(); i++)
      {
        String name = indexes.get(i).toString();
        double value = 1.0;
        if (!ignore_value)
          value = ((Number) Values.get(i)).doubleValue();
        keys[i] = name;
        values[i] = value;
      }
    }
    if (!loglik)
    {
      return o + eval(keys, values, num_click_replicates);
    }
    else
    {
      double xbeta = o + eval(keys, values, num_click_replicates);
      if (y == 1)
      {
        return -Math.log(1 + Math.exp(-xbeta));
      }
      else
      {
        return -Math.log(1 + Math.exp(xbeta));
      }
    }
  }

  public double evalInstanceJSON(Map<String, ?> input,
                                 int schema_type,
                                 boolean loglik,
                                 boolean ignore_value) throws IOException
  {
    return evalInstanceJSON(input, schema_type, loglik, 1, ignore_value);
  }

  public double evalFeatureAvro(List features, boolean ignore_value) throws IOException
  {
    int m = features.size();
    String[] keys = new String[m];
    double[] values = new double[m];
    for (int i = 0; i < m; i++)
    {
      Object temp = features.get(i);
      if (!(temp instanceof GenericData.Record))
        throw new IOException("features[" + i + "] is not a Avro Record");
      GenericData.Record feature = (GenericData.Record) temp;
      String name = Util.getStringAvro(feature, "name", false);
      String term = Util.getStringAvro(feature, "term", true);
      if (!"".equals(term))
        name = name + "\u0001" + term;
      double value = 1.0;
      if (!ignore_value)
        value = Util.getDoubleAvro(feature, "value");
      keys[i] = name;
      values[i] = value;
    }
    return eval(keys, values, 1);
  }

  /**
   * 
   * @param input
   * @param loglik
   * @param num_click_replicates
   * @param ignore_value
   * @return
   * @throws IOException
   */
  public double evalInstanceAvro(GenericData.Record input,
                                 boolean loglik,
                                 int num_click_replicates,
                                 boolean ignore_value) throws IOException
  {
    // response
    int y = Util.getResponseAvro(input);
    if (y != 1 && y != 0 && y != -1)
      throw new IOException("response = " + y);
    // offset
    double o = 0;
    if (input.get("offset") != null)
      o = Util.getDoubleAvro(input, "offset");
    // weight
    double weight = 1;
    if (input.get("weight") != null)
    {
      weight = Util.getDoubleAvro(input, "weight");
    }
    // features
    String[] keys = null;
    double[] values = null;
    Object temp = input.get("features");
    if (temp == null)
      throw new IOException("features is null");
    if (!(temp instanceof List))
      throw new IOException("features is not a list");
    List<?> features = (List<?>) temp;
    int m = features.size();
    keys = new String[m];
    values = new double[m];
    for (int i = 0; i < m; i++)
    {
      temp = features.get(i);
      if (!(temp instanceof GenericData.Record))
        throw new IOException("features[" + i + "] is not a Avro Record");
      GenericData.Record feature = (GenericData.Record) temp;
      String name = Util.getStringAvro(feature, "name", false);
      String term = Util.getStringAvro(feature, "term", true);
      if (!"".equals(term))
        name = name + "\u0001" + term;
      double value = 1.0;
      if (!ignore_value)
        value = Util.getDoubleAvro(feature, "value");
      keys[i] = name;
      values[i] = value;
    }
    if (!loglik)
    {
      return o + eval(keys, values, num_click_replicates);
    }
    else
    {
      double xbeta = o + eval(keys, values, num_click_replicates);
      if (y == 1)
      {
        return -Math.log1p(Math.exp(-xbeta)) * weight;
      }
      else
      {
        return -Math.log1p(Math.exp(xbeta)) * weight;
      }
    }
  }
  /**
   * 
   * @param input
   * @param loglik
   * @param num_click_replicates
   * @param ignore_value
   * @return
   * @throws IOException
   */
  public double evalInstanceAvro(GenericData.Record input,
                                 boolean loglik,
                                 boolean ignore_value) throws IOException
  {
    return evalInstanceAvro(input, loglik, 1, ignore_value);
  }
  /**
   * Output a string with format like "key1=value1 key2=value2 ...". intercept_key is the
   * key for intercept in the output string, such as "0". InnerDelim is the feature-value
   * delimiter ("=") and OuterDelim is the delimiter between features(",").
   * 
   * @param intercept_key
   * @param InnerDelim
   * @param OuterDelim
   * @return
   */
  public String toString(String intercept_key, String InnerDelim, String OuterDelim)
  {
    String output = intercept_key + InnerDelim + String.valueOf(_intercept);
    Iterator<String> iter = _coefficients.keySet().iterator();
    while (iter.hasNext())
    {
      String key = iter.next();
      output = output + OuterDelim;
      output = output + key + InnerDelim + String.valueOf(_coefficients.get(key));
    }
    return output;
  }

  /**
   * Output a string with format like "key1=value1 key2=value2,...". intercept_key is the
   * key for intercept in the output string, such as "0".
   * 
   * @param intercept_key
   * @return
   */
  public String toString(String intercept_key)
  {
    return this.toString(intercept_key, "=", " ");
  }

  /**
   * Output a string with format like "key1=value1 key2=value2,...". NOTE: intercept_key
   * is "0"!
   */
  public String toString()
  {
    return this.toString("0", "=", " ");
  }

  public double getIntercept()
  {
    return _intercept;
  }

  public void setIntercept(double value)
  {
    _intercept = value;
  }

  public Map<String, Double> getCoefficients()
  {
    return _coefficients;
  }

  public void setCoefficients(Map<String, Double> coefficients)
  {
    _coefficients.putAll(coefficients);
  }

  /**
   * Copy a linear model to a new model.
   * 
   * @param model
   * @return
   */
  public LinearModel copy()
  {
    LinearModel newmodel = new LinearModel();
    newmodel.setIntercept(_intercept);
    newmodel.setCoefficients(_coefficients);
    return newmodel;
  }

  /**
   * Convert a Linear model to a Map <String, Double>
   * 
   * @return
   */
  public Map<String, Double> toMap(String intercept_key)
  {
    Map<String, Double> map = new HashMap<String, Double>();
    map.putAll(_coefficients);
    map.put(intercept_key, _intercept);
    // DEBUG begin
    // for(String k : _coefficients.keySet()){
    // if(k == null) throw new RuntimeException("some key is null");
    // }
    // DEBUG end
    return map;
  }

  /**
   * Convert the model to a list of <name,term,value>
   * 
   * @param intercept_key
   * @return
   */
  public List<Map<String, Object>> toList(String intercept_key)
  {
    List<Map<String, Object>> list = new ArrayList<Map<String, Object>>();
    Map<String, Object> intercept_map = new HashMap<String, Object>();
    intercept_map.put("name", intercept_key);
    intercept_map.put("term", "");
    intercept_map.put("value", (float) _intercept);
    list.add(intercept_map);
    for (String k : _coefficients.keySet())
    {
      String[] token = k.split("\u0001");
      String name = token[0];
      String term = "";
      if (token.length > 1)
        term = token[1];
      Map<String, Object> map = new HashMap<String, Object>();
      map.put("name", name);
      map.put("term", term);
      double value = _coefficients.get(k);
      map.put("value", (float) value);
      list.add(map);
    }
    return list;
  }

  public List<GenericData.Record> toAvro(String intercept_key)
  {
    List<GenericData.Record> list = new ArrayList<GenericData.Record>();
    GenericData.Record intercept_record = new GenericData.Record(feature.SCHEMA$);
    intercept_record.put("name", intercept_key);
    intercept_record.put("term", "");
    intercept_record.put("value", (float) _intercept);
    list.add(intercept_record);
    for (String k : _coefficients.keySet())
    {
      String[] token = k.split("\u0001");
      String name = token[0];
      String term = "";
      if (token.length > 1)
        term = token[1];
      GenericData.Record record = new GenericData.Record(feature.SCHEMA$);
      record.put("name", name);
      record.put("term", term);
      double value = _coefficients.get(k);
      record.put("value", (float) value);
      list.add(record);
    }
    return list;
  }

  /**
   * Returns the maximum absolute value of this model use for compute convergence for
   * debugging purpose
   * 
   * @return
   */
  public double maxAbsValue()
  {
    double maxabs = Math.abs(_intercept);
    for (String k : _coefficients.keySet())
    {
      double v = Math.abs(_coefficients.get(k));
      if (maxabs < v)
      {
        maxabs = v;
      }
    }
    return maxabs;
  }

  /**
   * Filter out coefficients that contains String key in the coefficients
   * 
   * @param key
   */
  public void filterout(String key)
  {
    List<String> filter_keys = new ArrayList<String>();
    for (String k : _coefficients.keySet())
    {
      if (k.contains(key))
        filter_keys.add(k);
    }
    for (String k : filter_keys)
    {
      _coefficients.remove(k);
    }
  }

  /**
   * Remove all coefficients from this linear model. intercept will be reset to 0
   */
  public void clear()
  {
    _coefficients.clear();
    _intercept = 0;
  }
}
