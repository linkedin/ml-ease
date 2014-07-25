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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileStream;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.io.DatumReader;
import org.apache.avro.mapred.AvroInputFormat;
import org.apache.avro.mapred.AvroOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.log4j.Logger;


public class AvroUtils
{
  private static final Logger _log = Logger.getLogger(AvroUtils.class);
  /**
   * Adds all subdirectories under a root path to the input format.
   * 
   * @param conf The JobConf.
   * @param path The root path.
   * @throws IOException
   */
  public static void addAllSubPaths(JobConf conf, Path path) throws IOException
  {
    if (shouldPathBeIgnored(path)) 
    {
      throw new IllegalArgumentException(String.format("Path[%s] should be ignored.", path));
    }

    final FileSystem fs = path.getFileSystem(conf);

    if(fs.exists(path)) 
    {
      for (FileStatus status : fs.listStatus(path)) 
      {
        if (! shouldPathBeIgnored(status.getPath())) 
        {
          if (status.isDir()) 
          {
            addAllSubPaths(conf, status.getPath());
          }
          else 
          {
            AvroInputFormat.addInputPath(conf, status.getPath());
          }
        }
      }
    }
  }
  /**
   * Enumerates all the files under a given path.
   * 
   * @param conf The JobConf.
   * @param basePath The base path.
   * @return A list of files found under the base path.
   * @throws IOException
   */
  public static List<Path> enumerateFiles(JobConf conf, Path basePath) throws IOException
  {
    if (shouldPathBeIgnored(basePath)) 
    {
      throw new IllegalArgumentException(String.format("Path[%s] should be ignored.", basePath));
    }
    List<Path> paths = new ArrayList<Path>();
    FileSystem fs = basePath.getFileSystem(conf);
    
    if (!fs.exists(basePath))
    {
      return Collections.emptyList();
    }
    
    for (FileStatus s: fs.listStatus(basePath))
    {
      if (! shouldPathBeIgnored(s.getPath())) 
      {
        if (s.isDir())
        {
          paths.addAll(enumerateFiles(conf, s.getPath()));
        }
        else
        {
          paths.add(s.getPath());
        }
      }
    }
    return paths;
  }
  /**
   * Check if the path should be ignored. Currently only paths with "_log" are ignored.
   * 
   * @param path
   * @return
   * @throws IOException
   */
  public static boolean shouldPathBeIgnored(Path path) throws IOException
  {
    return path.getName().startsWith("_");
  }
  
  /**
   * Loads the schema from an Avro data file.
   * 
   * @param conf The JobConf.
   * @param path The path to the data file.
   * @return The schema read from the data file's metadata.
   * @throws IOException
   */
  public static Schema getSchemaFromFile(JobConf conf, Path path) throws IOException
  {
    FileSystem fs = path.getFileSystem(new Configuration());
    FSDataInputStream dataInputStream = fs.open(path);
    DatumReader <GenericRecord> reader = new GenericDatumReader<GenericRecord>();
    DataFileStream<GenericRecord> dataFileStream = new DataFileStream<GenericRecord>(dataInputStream, reader);
    return dataFileStream.getSchema();
  }
  
  /**
   * Given a path to an output folder, it finds the existing "*.avro" files and adds 
   * them as cache files to be distributed. Throws an exception if no files are found/added.
   * 
   * @param conf Job configuration
   * @param outPath The path to the hdfs directory that has part files to cache
   * @throws Exception If no file is found at outPath throws a RuntimeException 
   */
  public static void addAvroCacheFiles(JobConf conf, Path outPath) throws Exception
  {
     FileStatus[] partFiles = getAvroPartFiles(conf, outPath);
     if (partFiles.length == 0)
     {      
       throw new RuntimeException("DistributedCacheFileUtils: No (part) file is found to cache at location:" + outPath );
     }
     
     for (FileStatus partFile : partFiles)
     {
       // add the file and set fileRead to true, since we have read at least one file
       DistributedCache.addCacheFile(partFile.getPath().toUri(), conf);
     }
   }
  
  public static FileStatus[] getAvroPartFiles(JobConf conf, Path outPath) throws IOException
  {
    Path outputPath = outPath;
    FileSystem fileSystem = outputPath.getFileSystem(conf);

    FileStatus[] partFiles = fileSystem.listStatus(outputPath, new PathFilter()
    {
      @Override
      public boolean accept(Path path)
      {
        if (path.getName().endsWith(".avro"))
        {
          return true;
        }
        return false;
      }
    });
    
    return partFiles;
  }
  /**
   * Obtain the avro input schema from data
   * @param conf
   * @return
   * @throws IOException
   */
  public static Schema getAvroInputSchema(JobConf conf) throws IOException
  {
    Path[] paths = FileInputFormat.getInputPaths(conf);
    if (paths == null)
    {
      throw new IllegalStateException("input paths do not exist in jobConf!");
    }
    Schema inputSchema = AvroUtils.getSchemaFromFile(conf, paths[0]);
    if (inputSchema == null)
    {
      throw new IllegalStateException("Input does not have schema info and/or input is missing.");
    }
    return inputSchema;
  }
  
  /**
   * Run an avro hadoop job with job conf
   * @param conf
   * @throws Exception
   */
  public static void runAvroJob(JobConf conf) throws Exception
  {
    Path[] inputPaths = AvroInputFormat.getInputPaths(conf);
    _log.info("Running hadoop job with input paths:");
    for (Path inputPath : inputPaths)
    {
      _log.info(inputPath);
    }
    _log.info("Output path="+AvroOutputFormat.getOutputPath(conf));
    Job job = new Job(conf);
    job.setJarByClass(AvroUtils.class);
    job.waitForCompletion(true);
  }
  
  /**
   * Obtain a DataFileStream given a conf and path
   * @param conf
   * @param path
   * @return
   * @throws IOException
   */
  public static DataFileStream<Object> getAvroDataStream(JobConf conf, Path path) throws IOException
  {
    FileSystem fs = path.getFileSystem(conf);
    GenericDatumReader<Object> avroReader = new GenericDatumReader<Object>();
    InputStream hdfsInputStream = fs.open(path);
    return new DataFileStream<Object>(hdfsInputStream, avroReader);
  }
  public static void addAvroCacheFilesAndSetTheProperty(JobConf conf,
                                                          Path inputPath,
                                                          String property) throws Exception
  {
    addAvroCacheFiles(conf, inputPath);
    conf.set(property, inputPath.toString());
  }
}
