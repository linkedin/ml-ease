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

public final class ReadLambdaRhoConsumer implements AvroConsumer<Map<Float, Float>>
{
  private Map<Float, Float> _result = new HashMap<Float, Float>();
  private boolean           _done   = false;

  @Override
  public void consume(Object value)
  {
    GenericData.Record record = (GenericData.Record) value;
    float lambda = (Float) record.get("lambda");
    float rho = (Float) record.get("rho");
    _result.put(lambda, rho);
  }

  @Override
  public void done()
  {
    _done = true;
  }

  @Override
  public Map<Float, Float> get() throws IllegalStateException
  {
    if (_done)
    {
      return _result;
    }
    throw new IllegalStateException("Cannot call get before done");
  }
}
