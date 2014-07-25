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

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.generic.GenericDatumWriter;
import org.apache.avro.io.DatumWriter;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

public class AvroHdfsFileWriter <T>
{
  private DataFileWriter <T> _recordWriter;
  
  public AvroHdfsFileWriter (JobConf conf, String path, Schema schema) throws IOException
  {
        FileSystem fs = FileSystem.get(conf);
        FSDataOutputStream out = fs.create(new Path(path));
        DatumWriter<T> writer = new GenericDatumWriter<T>(schema);
        _recordWriter = new DataFileWriter<T>(writer);
        _recordWriter.create(schema, out);
  }
  public DataFileWriter <T> get()
  {
      return _recordWriter;
  }
}
