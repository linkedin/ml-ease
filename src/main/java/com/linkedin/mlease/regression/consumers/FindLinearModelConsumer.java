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
import java.util.List;

import org.apache.avro.generic.GenericData;

import com.linkedin.mapred.AvroConsumer;
import com.linkedin.mlease.models.LinearModel;
import com.linkedin.mlease.utils.Util;

public final class FindLinearModelConsumer implements AvroConsumer<LinearModel>
{
  private LinearModel        _result       = new LinearModel();
  private boolean            _done         = false;
  private String             _param;
  public static final String INTERCEPT_KEY = "(INTERCEPT)";

  public FindLinearModelConsumer(String param)
  {
    _param = param;
  }

  @Override
  public void consume(Object value)
  {
    GenericData.Record record = (GenericData.Record) value;
    if (record.get("model") != null && record.get("key") != null)
    {
      try
      {
        String partitionID = Util.getStringAvro(record, "key", false);
        if (partitionID.equals(_param))
        {
          _result = new LinearModel(INTERCEPT_KEY, (List<?>) record.get("model"));
          _done = true;
        }
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
  public LinearModel get() throws IllegalStateException
  {
    if (_done)
    {
      return _result;
    }
    throw new IllegalStateException("Cannot call get before done");
  }
}
