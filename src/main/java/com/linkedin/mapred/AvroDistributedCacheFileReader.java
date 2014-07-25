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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

public class AvroDistributedCacheFileReader extends AvroFileReader
{
  public AvroDistributedCacheFileReader(JobConf conf, boolean isSpecific)
  {
    super(conf, isSpecific);
  }
  
  public AvroDistributedCacheFileReader(JobConf conf)
  {
    this(conf, false);
  }

  @Override
  protected List<Path> getPaths(String filePath) throws IOException
  {
    Path[] localFiles = DistributedCache.getLocalCacheFiles(getConf());
    List<Path> paths = new ArrayList<Path>();
    
    for (Path file: localFiles)
    {
      if (!file.toString().contains(filePath))
      {
        continue;
      }
      
      paths.add(file);
    }
      
    return paths;
  }

  @Override
  protected FileSystem getFilesystem(JobConf conf, Path path) throws IOException
  {
    return FileSystem.getLocal(new Configuration());
  }
}

