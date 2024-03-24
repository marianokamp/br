CREATE EXTERNAL TABLE run_reports ( 
    region string,
    prompt string,
    completion string,
    actual_model string,
    client_measured_time_to_first_token_s double,
    metrics struct<
        inputTokenCount: integer,
        outputTokenCount: integer,
        invocationlatency: integer,
        firstByteLatency: integer>,
    client_measured_latency_s double,
    scenario string,
    max_tokens integer,
    model_id string,
    executed_at timestamp
)   
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe' 
LOCATION 's3://mkamp-aws-dub/run_reports/'


SELECT 
    scenario,
    region,
    actual_model as model,

    executed_at, 
    year(executed_at) as year,
    month(executed_at) as month,
    hour(executed_at) as hour,
    minute(executed_at) as minute,
    
    metrics.inputtokencount   as input_token_count,
    metrics.outputtokencount as output_token_count,
    
    metrics.firstbytelatency/1000. as server_first_byte_latency_s,
    metrics.invocationlatency/1000. as server_invocation_latency_s,
    
    client_measured_latency_s as client_invocation_latency_s,
    client_measured_time_to_first_token_s as server_first_token_latency_s
    
    
FROM "default"."run_reports"

