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
import java.net.URISyntaxException;
import java.util.List;

import org.apache.avro.Schema;
import org.apache.avro.mapred.AvroInputFormat;
import org.apache.avro.mapred.AvroJob;
import org.apache.avro.mapred.AvroMapper;
import org.apache.avro.mapred.AvroOutputFormat;
import org.apache.avro.mapred.AvroReducer;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public abstract class AbstractAvroJob
{
  public static final String HADOOP_PREFIX = "hadoop-conf.";
  public static final String INPUT_PATHS   = "input.paths";
  public static final String OUTPUT_PATH   = "output.path";
  private static final Logger _log = Logger.getLogger(AbstractAvroJob.class);
  private String _jobId;
  private final JobConfig _config;
  
  protected AbstractAvroJob(String jobId, JobConfig config)
  {
    _jobId = jobId;
    _config = config;
    Level loggingLevel = Level.toLevel(_config.getString("logging.level", "DEBUG"));
    _log.setLevel(loggingLevel);
  }
  
  protected AbstractAvroJob(JobConfig config)
  {
    this("ADMM-"+String.valueOf((int)(Math.random()*100000000)), config);
    try
    {
      _jobId = config.getString("job_name");
    } catch (Exception e)
    {
      _log.info("Can't find job_name key in the config, set job_name to be " + _jobId);
    }
  }
  
  public abstract void run() throws Exception;
  
  /**
   * Creates a JobConf for a map-only job. Automatically loads the schema from each input file.
   * 
   * @param mapperClass AvroMapper subclass implementing the map phase
   * @param outputSchema Schema of the mapper output
   * @return A configured JobConf.
   * @throws IOException
   * @throws URISyntaxException 
   */
  public JobConf createJobConf(Class<? extends AvroMapper> mapperClass, 
                               Schema outputSchema) throws IOException, URISyntaxException
  {
    JobConf conf = createJobConf();

    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, AvroReducer.class);

    AvroJob.setOutputSchema(conf, outputSchema);
    
    conf.setNumReduceTasks(0);

    return conf;
  }

  /**
   * Creates a JobConf for a map-reduce job. Loads the input schema from the input files.
   * 
   * @param mapperClass AvroMapper subclass for the mapper.
   * @param reducerClass AvroReducer subclass for the reducer.
   * @param mapperOutputSchema Mapper output schema. Must be an instance of org.apache.avro.mapred.Pair
   * @param outputSchema Reducer output schema
   * @return A configured JobConf.
   * @throws IOException
   * @throws URISyntaxException 
   */
  public JobConf createJobConf(Class<? extends AvroMapper> mapperClass,
                               Class<? extends AvroReducer> reducerClass,
                               Schema mapperOutputSchema,
                               Schema outputSchema) throws IOException, URISyntaxException
  {
    JobConf conf = createJobConf();

    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, reducerClass);

    AvroJob.setMapOutputSchema(conf, mapperOutputSchema);
    AvroJob.setOutputSchema(conf, outputSchema);

    return conf;
  }  

  /**
   * Creates a JobConf for a map-reduce job that uses a combiner. Loads the input schema from the
   * input files.
   * 
   * @param mapperClass AvroMapper subclass for the mapper.
   * @param reducerClass AvroReducer subclass for the reducer.
   * @param combinerClass AvroReducer subclass for the combiner.
   * @param mapperOutputSchema Mapper output schema. Must be an instance of org.apache.avro.mapred.Pair
   * @param outputSchema Reducer output schema
   * @return A configured JobConf.
   * @throws IOException
   * @throws URISyntaxException 
   */
  public JobConf createJobConf(Class<? extends AvroMapper> mapperClass,
                               Class<? extends AvroReducer> reducerClass,
                               Class<? extends AvroReducer> combinerClass,
                               Schema mapperOutputSchema,
                               Schema outputSchema) throws IOException, URISyntaxException
  {
    JobConf conf = createJobConf();
    
    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, reducerClass);
    AvroJob.setCombinerClass(conf, combinerClass);
    
    AvroJob.setMapOutputSchema(conf, mapperOutputSchema);
    AvroJob.setOutputSchema(conf, outputSchema);
    
    return conf;
  }
  
  /**
   * Creates a JobConf for a map-only job with an explicitly set input Schema.
   * 
   * @param mapperClass AvroMapper subclass implementing the map phase
   * @param inputSchema Schema of the input data.
   * @param outputSchema Schema of the mapper output
   * @return A configured JobConf.
   * @throws IOException
   * @throws URISyntaxException 
   */
  public JobConf createJobConf(Class<? extends AvroMapper> mapperClass, 
                               Schema inputSchema, 
                               Schema outputSchema) throws IOException, URISyntaxException
  {
    JobConf conf = createJobConf();

    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, AvroReducer.class);
    
    AvroJob.setInputSchema(conf, inputSchema);
    AvroJob.setOutputSchema(conf, outputSchema);
    
    conf.setNumReduceTasks(0);

    return conf;
  }
  
  /**
   * Creates a JobConf for a map-reducer job with an explicitly set input schema.
   * 
   * @param mapperClass AvroMapper subclass for the mapper.
   * @param reducerClass AvroReducer subclass for the reducer.
   * @param inputSchema Schema of the input data.
   * @param mapperOutputSchema Mapper output schema. Must be an instance of org.apache.avro.mapred.Pair
   * @param outputSchema Reducer output schema
   * @return A configured JobConf.
   * @throws IOException
   * @throws URISyntaxException 
   */
  public JobConf createJobConf(Class<? extends AvroMapper> mapperClass,
                               Class<? extends AvroReducer> reducerClass,
                               Schema inputSchema,
                               Schema mapperOutputSchema,
                               Schema outputSchema) throws IOException, URISyntaxException
  {
    JobConf conf = createJobConf();
    
    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, reducerClass);
    
    AvroJob.setInputSchema(conf, inputSchema);
    AvroJob.setMapOutputSchema(conf, mapperOutputSchema);
    AvroJob.setOutputSchema(conf, outputSchema);

    return conf;
  }

  /**
   * Creates a JobConf for a map-reduce job that uses a combiner and has an explicitly set input schema.
   * 
   * @param mapperClass AvroMapper subclass for the mapper.
   * @param reducerClass AvroReducer subclass for the reducer.
   * @param combinerClass AvroReducer subclass for the combiner.
   * @param inputSchema Schema of the input data.
   * @param mapperOutputSchema Mapper output schema. Must be an instance of org.apache.avro.mapred.Pair
   * @param outputSchema Reducer output schema
   * @return A configured JobConf.
   * @throws IOException
   * @throws URISyntaxException 
   */
  public JobConf createJobConf(Class<? extends AvroMapper> mapperClass,
                               Class<? extends AvroReducer> reducerClass,
                               Class<? extends AvroReducer> combinerClass,
                               Schema inputSchema,
                               Schema mapperOutputSchema,
                               Schema outputSchema) throws IOException, URISyntaxException
  {
    JobConf conf = createJobConf();
    
    AvroJob.setMapperClass(conf, mapperClass);
    AvroJob.setReducerClass(conf, reducerClass);
    AvroJob.setCombinerClass(conf, combinerClass);
    
    AvroJob.setInputSchema(conf, inputSchema);
    AvroJob.setMapOutputSchema(conf, mapperOutputSchema);
    AvroJob.setOutputSchema(conf, outputSchema);

    return conf;
  }
  
  /**
   * Sets up various standard settings in the JobConf. You probably don't want to mess with this.
   * 
   * @return A configured JobConf.
   * @throws IOException
   * @throws URISyntaxException 
   */
  protected  JobConf createJobConf() throws IOException, URISyntaxException
  {
    JobConf conf = new JobConf();
    
    conf.setJobName(getJobId());
    conf.setInputFormat(AvroInputFormat.class);
    conf.setOutputFormat(AvroOutputFormat.class);
    
    AvroOutputFormat.setDeflateLevel(conf, 9);
    
    String hadoop_ugi = _config.getString("hadoop.job.ugi", null);
    if (hadoop_ugi != null)
    {
        conf.set("hadoop.job.ugi", hadoop_ugi);
    }
    if (_config.getBoolean("is.local", false))
    {
      conf.set("mapred.job.tracker", "local");
      conf.set("fs.default.name", "file:///");
      conf.set("mapred.local.dir", "/tmp/map-red");

      _log.info("Running locally, no hadoop jar set.");
    }
    
    // set JVM options if present
    if (_config.containsKey("mapred.child.java.opts"))
    {
      conf.set("mapred.child.java.opts", _config.getString("mapred.child.java.opts"));
      _log.info("mapred.child.java.opts set to " + _config.getString("mapred.child.java.opts"));
    }

    if (_config.containsKey(INPUT_PATHS))
    {
      List<String> inputPathnames = _config.getStringList(INPUT_PATHS);
      for (String pathname : inputPathnames)
      {
        AvroUtils.addAllSubPaths(conf, new Path(pathname));
      }
      AvroJob.setInputSchema(conf, AvroUtils.getAvroInputSchema(conf));
    }

    if (_config.containsKey(OUTPUT_PATH))
    {
      Path path = new Path(_config.get(OUTPUT_PATH));
      AvroOutputFormat.setOutputPath(conf, path);

      if (_config.getBoolean("force.output.overwrite", false))
      {
        FileSystem fs = FileOutputFormat.getOutputPath(conf).getFileSystem(conf);
        fs.delete(FileOutputFormat.getOutputPath(conf), true);
      }
    }
    // set all hadoop configs
    for (String key : _config.keySet()) 
    {
      String lowerCase = key.toLowerCase();
      if ( lowerCase.startsWith(HADOOP_PREFIX)) 
      {
          String newKey = key.substring(HADOOP_PREFIX.length());
          conf.set(newKey, _config.get(key));
      }
    }
    return conf;
  }
  
  public String getJobId()
  {
    return _jobId;
  }
  
  public JobConfig getJobConfig()
  {
    return _config;
  }
}
