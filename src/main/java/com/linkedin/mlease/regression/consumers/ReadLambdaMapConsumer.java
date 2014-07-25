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

package com.linkedin.mlease.regression.consumers;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.avro.generic.GenericData;

import com.linkedin.mapred.AvroConsumer;
import com.linkedin.mlease.utils.Util;

public class ReadLambdaMapConsumer implements AvroConsumer<Map<String, Float>>
{
  private Map<String, Float> _result = new HashMap<String, Float>();
  private boolean            _done   = false;

  @Override
  public void consume(Object value)
  {
    GenericData.Record record = (GenericData.Record) value;
    if (record.get("name") != null && record.get("value") != null)
    {
      String name;
      try
      {
        name = Util.getStringAvro(record, "name", false);
        String term = Util.getStringAvro(record, "term", true);
        if (!"".equals(term))
          name = name + "\u0001" + term;
        float lambda = (float) Util.getDoubleAvro(record, "value");
        _result.put(name, lambda);
      }
      catch (IOException e)
      {
        e.printStackTrace();
      }
    }
  }

  @Override
  public void done()
  {
    _done = true;
  }

  @Override
  public Map<String, Float> get() throws IllegalStateException
  {
    if (_done)
    {
      return _result;
    }
    throw new IllegalStateException("Cannot call get before done");
  }
}
