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

public final class ReadLinearModelConsumer implements
    AvroConsumer<Map<String, LinearModel>>
{
  // result is a map of key:String => value:LinearModel
  private Map<String, LinearModel> _result       = new HashMap<String, LinearModel>();
  private boolean                  _done         = false;
  public static final String       INTERCEPT_KEY = "(INTERCEPT)";
  private String                   _lambdaStr;
  private int                      _partitionId;
  private int                      _nReducers;

  public ReadLinearModelConsumer()
  {
    _lambdaStr = null;
    _partitionId = -1;
    _nReducers = -1;
  }

  public ReadLinearModelConsumer(String lambdaStr, int partitionId, int nReducers)
  {
    _lambdaStr = lambdaStr;
    _partitionId = partitionId;
    _nReducers = nReducers;
  }

  @Override
  public void consume(Object value)
  {
    GenericData.Record record = (GenericData.Record) value;
    try
    {
      if (record.get("key") != null && record.get("model") != null)
      {
        String partitionID = Util.getStringAvro(record, "key", false);
        LinearModel model = new LinearModel(INTERCEPT_KEY, (List<?>) record.get("model"));
        if (_lambdaStr == null)
        {
          _result.put(partitionID, model);
        }
        else
        {
          if (partitionID.contains(_lambdaStr))
          {
            String[] token = partitionID.split("#");
            if (token.length > 1)
            {
              if (Math.abs(token[1].hashCode()) % _nReducers == _partitionId)
              {
                _result.put(partitionID, model);
              }
            }
          }
        }
      }
    }
    catch (IOException e)
    {
      e.printStackTrace();
    }
  }

  @Override
  public void done()
  {
    _done = true;
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
}
