import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class SOM extends Configured implements Tool{

	public static class Map extends MapReduceBase implements Mapper<LongWritable,Text,IntWritable,Text>{
		static enum Counters {INPUT_INSTANCE}
		private static double[][] neuron;
		private Text word = new Text();
		int numberOfNeuron;
		int numberOfAttribute;
		double neighbourhoodSize;
		double decay;

		public void configure(JobConf job){
			try{
				numberOfNeuron=Integer.parseInt(job.get("numOfNeuron"));
				numberOfAttribute=Integer.parseInt(job.get("attributeSize"));
				neighbourhoodSize=Double.parseDouble(job.get("neighbourhoodSize"));
				decay=Double.parseDouble(job.get("decay"));
				Path[] initialFile = new Path[0];
				try{
					initialFile = DistributedCache.getLocalCacheFiles(job);
				}catch(IOException ioe){
					System.err.println("Caught exception while getting cached files: "+StringUtils.stringifyException(ioe));
				}
				for(Path weightFile : initialFile){
					parseInitialWeight(weightFile);
				}
			}catch(Exception ex){
				System.err.println("Error in getting needed parameter");
			}
		}

		private void parseInitialWeight(Path initialFile){
			try{
				BufferedReader br = new BufferedReader(new FileReader(initialFile.toString()));
				String line;
				StringTokenizer stk;
				int neuronNo = 0;
				double[] vector;
				neuron = new double[numberOfNeuron][numberOfAttribute];
				while((line = br.readLine())!=null){
					stk=new StringTokenizer(line);
					for(int i=0;i<numberOfAttribute;i++){
						neuron[neuronNo][i]=Double.parseDouble(stk.nextToken());
					}
					neuronNo++;
				}
				br.close();
			}catch(IOException ioe){
				System.err.println("Error in getting the initial weight file");
			}
		}

		private double computeDistance(double[] vector1,double[] vector2){
			double distance=0.0;
			for(int i=0;i<numberOfAttribute;i++){
				distance+=Math.pow((vector1[i]-vector2[i]),2.0);
			}
			distance=Math.sqrt(distance);
			return distance;
		}
		private int computeBMU(double[] vector){
			int bestNeuron=0;
			double bestDistance=Double.MAX_VALUE;
			double distance;
			for(int i=0;i<numberOfNeuron;i++){
				distance=computeDistance(vector,neuron[i]);
				if(distance < bestDistance){
					bestDistance=distance;
					bestNeuron=i;
				}
			}
			return bestNeuron;
		}

		public void map(LongWritable key,Text value,OutputCollector<IntWritable,Text> output,Reporter reporter) throws IOException{
			String line = value.toString();
			StringTokenizer idata = new StringTokenizer(line,",");
			double[] vector = new double[numberOfAttribute];
			for(int i=0;i<numberOfAttribute;i++){
				vector[i]=Double.parseDouble(idata.nextToken());
			}
			int BMU = computeBMU(vector);
			double tempHck;
			double tempWidth;
			double tempUp;

			for(int j=0;j<numberOfNeuron;j++){
				tempUp=-(Math.pow((computeDistance(neuron[BMU],neuron[j])),2.0));
				tempWidth=2 * Math.pow((neighbourhoodSize * Math.exp((-(double)((reporter.getCounter(Counters.INPUT_INSTANCE)).getValue())/decay))),2.0);
				tempHck=Math.exp(tempUp/tempWidth);
				word.clear();
				word.set(Double.toString(tempHck));
				output.collect(new IntWritable(j), word);
				StringBuilder sb = new StringBuilder();
				sb.append("@");
				for(int k=0;k<numberOfAttribute;k++){
					sb.append(vector[k]*tempHck);
					if(k!=(numberOfAttribute-1)){
						sb.append(',');
					}
				}
				word.clear();
				word.set(sb.toString());
				output.collect(new IntWritable(j), word);
			}
			reporter.incrCounter(Counters.INPUT_INSTANCE, 1);
		}
	}

	public static class Reduce extends MapReduceBase implements Reducer<IntWritable,Text,IntWritable,Text>{
		int numberOfAttribute;
		private Text word = new Text();

		public void configure(JobConf job){
			try{
				numberOfAttribute=Integer.parseInt(job.get("attributeSize"));
			}catch(Exception ex){
				System.err.println("Error in getting needed parameter");
			}
		}

		public void reduce(IntWritable key,Iterator<Text> values,OutputCollector<IntWritable,Text> output,Reporter reporter) throws IOException{
			double hck = 0.0;
			double[] hckX = new double[numberOfAttribute];
			String weight;
			String head;
			StringTokenizer idata;
			int i;
			// initialize hckX
			for(i=0;i<numberOfAttribute;i++){
				hckX[i]=0.0;
			}
			int count=0;
			while(values.hasNext()){
				weight=values.next().toString();
				head=weight.substring(0,1);
				if(head.equals("@")){
					weight=weight.substring(1);
					idata=new StringTokenizer(weight,",");
					for(i=0;i<numberOfAttribute;i++){
						hckX[i]+=Double.parseDouble(idata.nextToken());
					}
				}else{
					hck+=Double.parseDouble(weight);
				}
			}

			StringBuilder sb = new StringBuilder();
			for(i=0;i<numberOfAttribute;i++){
				sb.append(hckX[i]/hck);
				if(i!=(numberOfAttribute-1)){
					sb.append(",");
				}
			}
			word.clear();
			word.set(sb.toString());
			output.collect(key, word);

		}
	}


	public int run(String[] args) throws Exception{
		JobConf conf = new JobConf(getConf(),SOM.class);
		conf.setJobName("SOM");

		conf.setOutputKeyClass(IntWritable.class);
		conf.setOutputValueClass(Text.class);

		conf.setMapperClass(Map.class);
		conf.setReducerClass(Reduce.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

		if(args.length!=7){
			System.err.println("bin/hadoop jar SOM.jar SOM [inputFile] [outputFolder] [# of Neuron] [# of attributes] [neighbourhood size] [decay] [initial weight file]");
			return 0;
		}else{

			FileInputFormat.setInputPaths(conf,new Path(args[0]));
			FileOutputFormat.setOutputPath(conf,new Path(args[1]));

			conf.set("numOfNeuron",args[2]);
			conf.set("attributeSize",args[3]);
			conf.set("neighbourhoodSize",args[4]);
			conf.set("decay",args[5]);
			DistributedCache.addCacheFile(new Path(args[6]).toUri(),conf);

			JobClient.runJob(conf);
			return 0;
		}
	}

	public static void main(String[] args) throws Exception{
		int res = ToolRunner.run(new Configuration(),new SOM(),args);
		System.exit(res);
	}
}