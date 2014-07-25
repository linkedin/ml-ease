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

package com.linkedin.mlease.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.avro.Schema;
import org.apache.avro.generic.GenericData;
import org.apache.avro.util.Utf8;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobConf;

import com.linkedin.mapred.AvroUtils;

public class Util
{

  public static int getInt(Map<String, ?> map, String key) throws IOException
  {
    Object temp = map.get(key);
    if (temp == null)
      throw new IOException(key + " is null");
    if (!(temp instanceof Integer))
      throw new IOException(key + "=" + temp.toString() + " is not an integer");
    return ((Integer) temp).intValue();
  }

  public static int getIntAvro(GenericData.Record data, String key) throws IOException
  {
    Object temp = data.get(key);
    if (temp == null)
      throw new IOException(key + " is null");
    if (!(temp instanceof Integer))
      throw new IOException(key + "=" + temp.toString() + " is not an integer");
    return ((Integer) temp).intValue();
  }

  public static double getDouble(Map<String, ?> map, String key) throws IOException
  {
    Object temp = map.get(key);
    if (temp == null)
      throw new IOException(key + " is null");
    if (temp instanceof String)
    {
      return atof((String) temp);
    }
    if (!(temp instanceof Number))
      throw new IOException(key + "=" + temp.toString() + " is not a number");
    return ((Number) temp).doubleValue();
  }

  public static double getDoubleAvro(GenericData.Record data, String key) throws IOException
  {
    Object temp = data.get(key);
    if (temp == null)
      throw new IOException(key + " is null");
    if (temp instanceof String)
    {
      return atof((String) temp);
    }
    if (!(temp instanceof Number))
      throw new IOException(key + "=" + temp.toString() + " is not a number");
    return ((Number) temp).doubleValue();
  }

  public static boolean atob(String s)
  {
    if (s == null || s.length() < 1)
      throw new IllegalArgumentException("Cannot convert empty string to boolean");
    s = s.toLowerCase().trim();
    if (s.equals("true"))
      return true;
    if (s.equals("false"))
      return false;
    if (s.equals("1"))
      return true;
    if (s.equals("0"))
      return false;
    throw new IllegalArgumentException("Cannot convert '" + s + "' to boolean");
  }

  public static String getString(Map<String, ?> map, String key, boolean isNullOK) throws IOException
  {
    Object temp = map.get(key);
    if (temp == null)
    {
      if (isNullOK)
        return "";
      throw new IOException(key + " is null");
    }
    if (!(temp instanceof String))
      throw new IOException(key + "=" + temp.toString() + " is not a string");
    return ((String) temp);
  }

  public static String getStringAvro(GenericData.Record data, String key, boolean isNullOK) throws IOException
  {
    Object temp = data.get(key);
    if (temp == null)
    {
      if (isNullOK)
        return "";
      throw new IOException(key + " is null");
    }
    if (!(temp instanceof String || temp instanceof Utf8))
      throw new IOException(key + "=" + temp.toString() + " is not a string");
    return temp.toString();
  }

  /**
   * @param s
   *          the string to parse for the double value
   * @throws IllegalArgumentException
   *           if s is empty or represents NaN or Infinity
   * @throws NumberFormatException
   *           see {@link Double#parseDouble(String)}
   */
  public static double atof(String s)
  {
    if (s == null || s.length() < 1)
      throw new IllegalArgumentException("Can't convert empty string to double");
    double d = Double.parseDouble(s);
    if (Double.isNaN(d) || Double.isInfinite(d))
    {
      throw new IllegalArgumentException("NaN or Infinity in input: " + s);
    }
    return (d);
  }

  /**
   * @param s
   *          the string to parse for the integer value
   * @throws IllegalArgumentException
   *           if s is empty
   * @throws NumberFormatException
   *           see {@link Integer#parseInt(String)}
   */
  public static int atoi(String s) throws NumberFormatException
  {
    if (s == null || s.length() < 1)
      throw new IllegalArgumentException("Can't convert empty string to integer");
    // Integer.parseInt doesn't accept '+' prefixed strings
    if (s.charAt(0) == '+')
      s = s.substring(1);
    return Integer.parseInt(s);
  }

  /**
   * <p>
   * Each line in the file is in the following format: key=value (if separator is "=").
   * The leading and ending space will be removed.
   * </p>
   * <p>
   * Example:
   * </p>
   * 
   * <pre>
   * Name1 = 0.5
   * Name=abc=  0.8
   *   Name=1=10
   * </pre>
   * 
   * @return
   */
  public static Map<String, Double> readStringDoubleMap(File file, String separator) throws IOException
  {
    Map<String, Double> map = new HashMap<String, Double>();
    BufferedReader fp = new BufferedReader(new FileReader(file));
    String line = null;
    int line_no = 0;
    while ((line = fp.readLine()) != null)
    {
      line_no++;
      line = line.replaceFirst("\\s+$", "");
      line = line.replaceFirst("^\\s+", "");
      if (line.equals(""))
        continue;
      String[] token = line.split("\\s*" + separator + "\\s*");
      if (token.length < 2)
        throw new IOException("Format error in file '" + file.getPath() + "' at line "
            + line_no + ": " + line);
      double value = Double.parseDouble(token[token.length - 1]);
      int index = line.lastIndexOf(token[token.length - 1]);
      String name =
          line.substring(0, index).replaceFirst("\\s*" + separator + "\\s*$", "");
      map.put(name, value);
    }
    fp.close();
    return map;
  }

  public static void printStringDoubleMap(PrintStream out,
                                          Map<String, Double> map,
                                          String separator,
                                          boolean sort)
  {
    Object[] key = map.keySet().toArray();
    if (sort)
      Arrays.sort(key);
    for (int i = 0; i < key.length; i++)
    {
      out.println(key[i] + " = " + map.get(key[i]));
    }
  }

  public static void printStringListDoubleMap(PrintStream out,
                                          Map<List<String>, Double> map,
                                          String separator)
  {
    Object[] key = map.keySet().toArray();
    for (int i = 0; i < key.length; i++)
    {
      out.println(key[i].toString() + " = " + map.get(key[i]));
    }
  }

  public static String getLambda(String str)
  {
    String[] token = str.split("#");
    return token[0];
  }

  public static List<Path> findPartFiles(JobConf conf, Path root) throws IOException
  {
    FileSystem fs = root.getFileSystem(new JobConf());
    List<Path> files = new ArrayList<Path>();

    for (FileStatus status : fs.listStatus(root))
    {
      if (status.isDir())
      {
        files.addAll(findPartFiles(conf, status.getPath()));
      }
      else
      {
        files.add(status.getPath());
      }
    }
    return files;
  }

  public static boolean checkPath(String pathstr) throws IOException
  {
    if (pathstr.equals(""))
      return false;
    Path path = new Path(pathstr);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    return fs.exists(path);
  }

  public static int getResponse(Map<String, ?> map) throws IOException
  {
    Object response = null;
    if (map.containsKey("click"))
      response = map.get("click");
    if (map.containsKey("response"))
      response = map.get("response");
    if (map.containsKey("label"))
      response = map.get("label");
    if (response == null)
    {
      throw new IOException("Data should contain one field of the three: response, click or label!");
    }
    int new_response = 0;
    if (!(response instanceof Boolean) && (!(response instanceof Integer)))
    {
      throw new IOException("Response/Click/Label column should be either boolean or int32!");
    }
    if (response instanceof Boolean)
    {
      if ((Boolean) response)
        new_response = 1;
    }
    if (response instanceof Integer)
    {
      new_response = (Integer) response;
    }
    return new_response;
  }

  public static int getResponseAvro(GenericData.Record record) throws IOException
  {
    Object response = null;
    if (record.get("click") != null)
      response = record.get("click");
    if (record.get("response") != null)
      response = record.get("response");
    if (record.get("label") != null)
      response = record.get("label");
    if (response == null)
    {
      throw new IOException("Data should contain one field of the three: response, click or label!");
    }
    int new_response = 0;
    if (!(response instanceof Boolean) && (!(response instanceof Integer)))
    {
      throw new IOException("Response/Click/Label column should be either boolean or int32!");
    }
    if (response instanceof Boolean)
    {
      if ((Boolean) response)
        new_response = 1;
    }
    if (response instanceof Integer)
    {
      new_response = (Integer) response;
    }
    return new_response;
  }

  public static Schema removeUnion(Schema schema)
  {
    if (schema.getType() == Schema.Type.UNION)
    {
      List<Schema> schemas = schema.getTypes();
      for (Schema s : schemas)
      {
        if (s.getType() != Schema.Type.NULL)
        {
          return removeUnion(s);
        }
      }
    } else if (schema.getType() == Schema.Type.ARRAY)
    {
      Schema newSchema = Schema.createArray(removeUnion(schema.getElementType()));
      return newSchema;
    } else if (schema.getType() == Schema.Type.RECORD)
    {
      List<Schema.Field> fields = schema.getFields();
      List<Schema.Field> newFields = new LinkedList<Schema.Field>();
      for (Schema.Field f : fields)
      {
        newFields.add(new Schema.Field(f.name(),
                                       removeUnion(f.schema()),
                                       f.doc(),
                                       null));
      }
      Schema newSchema = 
          Schema.createRecord(schema.getName(),
                              schema.getDoc(),
                              schema.getNamespace(),
                              false);
      newSchema.setFields(newFields);
      return newSchema;
    } 
    return schema;
  }
}
