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
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

public class AvroHdfsFileReader extends AvroFileReader
{
  public AvroHdfsFileReader(JobConf conf, boolean isSpecific)
  {
    super(conf, isSpecific);
  }
  
  public AvroHdfsFileReader(JobConf conf)
  {
    this(conf, false);
  }
  
  private boolean isAvro(Path path)
  {
    return path.getName().endsWith(".avro");
  }
    
  @Override
  protected List<Path> getPaths(String filePath) throws IOException
  {
    Path path = new Path(filePath);
    FileSystem fs = path.getFileSystem(getConf());
    List<Path> paths = new ArrayList<Path>();
    
    for (FileStatus status: fs.listStatus(path))
    {
      if (status.isDir() && !AvroUtils.shouldPathBeIgnored(status.getPath()))
      {
        paths.addAll(getPaths(status.getPath().toString()));
      }
      else if (isAvro(status.getPath()))
      {
        paths.add(status.getPath());
      }
    }
    return paths;
  }

  @Override
  protected FileSystem getFilesystem(JobConf conf, Path path) throws IOException
  {
    return path.getFileSystem(conf);
  }  
}

