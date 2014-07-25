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

package com.linkedin.mapred;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.log4j.Logger;

public class JobConfig
{
  private static Logger logger = Logger.getLogger(JobConfig.class);
  private final Map<String, String> _configMap;
  
  public JobConfig()
  {
    _configMap = new ConcurrentHashMap<String, String>();
  }
  /**
   * load the job config from a file
   * @param file
   * @throws FileNotFoundException
   * @throws IOException
   */
  public JobConfig(String file) throws FileNotFoundException, IOException 
  {
    _configMap = getConfigMapFromFile(file);
  }
  /**
   * load the job config from a list of files
   * @param files
   * @throws FileNotFoundException
   * @throws IOException
   */
  public JobConfig(List<String> files) throws FileNotFoundException, IOException 
  {
    this();
    for (String file : files)
    {
      _configMap.putAll(getConfigMapFromFile(file));
    }
  }
  
  /**
   * load the job config from a Map<String, String>
   * @param configMap
   */
  public JobConfig(Map<String, String> configMap)
  {
    _configMap = new ConcurrentHashMap<String, String>(configMap);
  }
 
  private Map<String, String> getConfigMapFromFile(String file) throws FileNotFoundException, IOException 
  {
    Map<String, String> configMap = new HashMap<String, String>();
    InputStream input = new BufferedInputStream(new FileInputStream(new File(file).getAbsolutePath()));
    Properties properties = new Properties();
    properties.load(input);
    for(String propName: properties.stringPropertyNames()) 
    {
      configMap.put(propName, properties.getProperty(propName));
    }
    input.close();
    return configMap;
  }
  /**
   * Return the value given a key
   * @param key
   * @return
   */
  public String get(String key) 
  {
    return _configMap.get(key);
  }
  /**
   * Put a <key, value> pair into the map. Value is converted to a string by toString() method 
   */
  public void put(String key, Object value)
  {
    _configMap.put(key, value.toString());
  }
  /**
   * The get method that returns a string. If key doesn't exist in the map, return default value
   * @param key
   * @param defaultValue
   * @return
   */
  public String getString(String key, String defaultValue)
  {
    if (!containsKey(key))
    {
      return defaultValue;
    } else
    {
      return get(key);
    }
  }
  /**
   * The get method that returns a string. 
   * @param key
   * @param defaultValue
   * @return
   */
  public String getString(String key) throws IOException
  {
    if (!containsKey(key))
    {
      throw new IOException("The job config does not contain key = " + key);
    } else
    {
      return get(key);
    }
  }
  
  public int getInt(String key, int defaultValue)
  {
    if (!containsKey(key))
    {
      return defaultValue;
    } else
    {
      return Integer.parseInt(get(key));
    }
  }
  public int getInt(String key) throws IOException
  {
    if (!containsKey(key))
    {
      throw new IOException("The job config does not contain key = " + key);
    } else
    {
      return Integer.parseInt(get(key));
    }
  }
  public long getLong(String key, long defaultValue)
  {
    if (!containsKey(key))
    {
      return defaultValue;
    } else
    {
      return Long.parseLong(get(key));
    }
  }
  public float getFloat(String key, Number defaultValue)
  {
    if (!containsKey(key))
    {
      return defaultValue.floatValue();
    } else
    {
      return Float.parseFloat(get(key));
    }
  }
  public double getDouble(String key, Number defaultValue)
  {
    if (!containsKey(key))
    {
      return defaultValue.doubleValue();
    } else
    {
      return Double.parseDouble(get(key));
    }
  }
  public boolean getBoolean(String key, boolean defaultValue)
  {
    if (!containsKey(key))
    {
      return defaultValue;
    } else
    {
      return Boolean.parseBoolean(get(key));
    }
  }
  public List<String> getStringList(String key) throws IOException
  {
    return getStringList(key, "\\s*,\\s*");
  }

  public List<String> getStringList(String key, String sep) throws IOException
  {
    String value = get(key);
    if(value == null || value.trim().length() == 0)
        return Collections.emptyList();

    if(containsKey(key))
        return Arrays.asList(value.split(sep));
    else
        throw new IOException("Missing required property '" + key + "'");
  }
  public List<String> getStringList(String key, List<String> defaultValue, String sep) throws IOException
  {
    if(containsKey(key))
    {
        return getStringList(key, sep);
    }
    else
    {
        return defaultValue;
    }
  }
  public boolean containsKey(String key)
  {
    return _configMap.containsKey(key);
  }
  public Set<String> keySet()
  {
    return _configMap.keySet();
  }
  public Map<String, String> getConfigMap()
  {
    return _configMap;
  }
  public static JobConfig clone(JobConfig config)
  {
    return new JobConfig(config.getConfigMap());
  }
}
