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
import java.util.List;
import java.util.Map;

import org.apache.avro.generic.GenericData;

import com.linkedin.mapred.AvroConsumer;
import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.utils.Util;

public class MeanLinearModelConsumer implements AvroConsumer<Map<String, LinearModel>>
{
  // result is a map of key:String => value:LinearModel
  private Map<String, LinearModel> _result       = new HashMap<String, LinearModel>();
  private int                      _counter      = 0;
  private boolean                  _done         = false;
  private int                      _nblocks;
  public static final String       INTERCEPT_KEY = "(INTERCEPT)";

  public MeanLinearModelConsumer(int nblocks)
  {
    _nblocks = nblocks;
  }

  @Override
  public void consume(Object value)
  {
    GenericData.Record record = (GenericData.Record) value;
    int nblocks = _nblocks;
    if (record.get("model") != null && record.get("key") != null)
    {
      try
      {
        String partitionID = Util.getStringAvro(record, "key", false);
        String lambda = Util.getLambda(partitionID);
        if (!_result.containsKey(lambda))
        {
          _result.put(lambda, new LinearModel());
        }
        LinearModel model = _result.get(lambda);
        LinearModel newmodel =
            new LinearModel(INTERCEPT_KEY, (List<?>) record.get("model"));
        model.linearCombine(1.0, 1.0 / nblocks, newmodel);
        _result.put(lambda, model);
        _counter++;
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

  public boolean isDone()
  {
    return _done;
  }

  @Override
  public Map<String, LinearModel> get() throws IllegalStateException
  {
    if (_done)
    {
      return _result;
    }
    throw new IllegalStateException("Cannot call get before done");
  }

  public int getCounter()
  {
    return _counter;
  }
}
