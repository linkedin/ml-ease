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

/**
 *
 * Consumer interface for building structures from data in an avro file. An instance of this interface will
 * be passed to a file reader to build up an object from the contents of that file.
 * 
 * @param <T> The type of object that gets constructed.
 */
public interface AvroConsumer<T>
{
  /**
   * Called by the file reader for each value.
   * 
   * @param value The avro generic record for the row
   */
  public void consume(Object value);
  
  /**
   * Called after all values have been read from the input.
   */
  public void done();

  /**
   * @return The object built from the data read in.
   */
  public T get();
}

