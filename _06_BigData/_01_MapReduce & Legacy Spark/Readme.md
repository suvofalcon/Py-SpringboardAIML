## How to run the MRjob

python _01_RatingsBreakdown.py -r hadoop hdfs:///datasets/ml-100k/u.data --python-bin /home/suvo/opt/miniconda3/envs/pyspark/bin/python


### Trick

Create ~/.mrjob.conf with this configuration:

runners:
  hadoop:
    python_bin: /usr/local/bin/python3
    hadoop_bin: /usr/local/opt/hadoop/bin/hadoop
    hadoop_streaming_jar: /usr/local/opt/hadoop/libexec/share/hadoop/tools/lib/hadoop-streaming-2.8.0.jar

