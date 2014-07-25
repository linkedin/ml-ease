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

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import org.apache.avro.file.DataFileStream;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.specific.SpecificDatumReader;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

public abstract class AvroFileReader
{
  private final JobConf _conf;
  private final boolean _isSpecific;
  
  public AvroFileReader(JobConf conf)
  {
    this(conf, false);
  }
  
  public AvroFileReader(JobConf conf, boolean isSpecific)
  {
    _conf = conf;
    _isSpecific = isSpecific;
  }
  
  protected abstract List<Path> getPaths(String root) throws IOException;
  protected abstract FileSystem getFilesystem(JobConf conf, Path path) throws IOException;
  
  protected DataFileStream<Object> getAvroDataStream(Path path) throws IOException {
    FileSystem fs = getFilesystem(_conf, path);

    GenericDatumReader<Object> avroReader = _isSpecific ? new SpecificDatumReader<Object>() : new GenericDatumReader<Object>();
    InputStream hdfsInputStream = fs.open(path);
    return new DataFileStream<Object>(hdfsInputStream, avroReader);
}
  
  public <T> void build(String filePath, AvroConsumer<T> builder) throws IOException
  {
    
    List<Path> paths = getPaths(filePath);

    for (Path path: paths)
    {
      DataFileStream<Object> stream = null;
      try
      {
        stream = getAvroDataStream(path);
        while (stream.hasNext())
        {
          builder.consume(stream.next());
        }
      }
      finally
      {
        if (stream != null)
        {
          stream.close();
        }
      }
    }
    
    builder.done();
  }
  
  protected JobConf getConf()
  {
    return _conf;
  }
}

