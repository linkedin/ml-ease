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

import java.util.HashMap;
import java.util.Map;

import org.apache.avro.generic.GenericData;

import com.linkedin.mapred.AvroConsumer;

public class ReadPartitionIdAssignmentConsumer implements AvroConsumer<Map<String, Integer>>
{
  private Map<String, Integer> _result = new HashMap<String, Integer>();
  @Override
  public void consume(Object value)
  {
    GenericData.Record record = (GenericData.Record) value;
    if (record.get("key") != null && record.get("value") != null)
    {
      String key = record.get("key").toString();
      int partitionId = Integer.parseInt(record.get("value").toString());
      _result.put(key, partitionId);
    }
  }
  @Override
  public void done()
  {
  }

  @Override
  public Map<String, Integer> get() throws IllegalStateException
  {
    return _result;
  }
}
