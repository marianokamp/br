#!/usr/bin/env python

import logging
import time
import json
import argparse

from pprint import pprint
import boto3
from botocore.exceptions import ClientError

from config import Config

logger = logging.getLogger(__name__)

bedrock_admin = boto3.client(service_name="bedrock", region_name="us-east-1")


def list_models():
    response = bedrock_admin.list_foundation_models()
    models = response["modelSummaries"]
    model_ids = [m["modelId"] for m in models]

    print(model_ids)

    pprint(response)
    print("modelsIds:\n", "\n\t".join(model_ids))


def generate(
    prompt, model_id, max_tokens, region, print_fn=None, verbose=False, **kwargs
):
    # if not print_fn:
    #    print_fn = partial(print, end="")

    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)
    emd = dict(region="us-east-1", prompt=prompt, completion="")  # Execution metadata

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )
    started = time.time()
    try:
        response = bedrock.invoke_model_with_response_stream(
            body=body,
            modelId=model_id,
            # accept="application/json",
            # contentType="application/json",
        )  # , accept=accept, contentType=contentType)

        stream = response.get("body")
        if verbose:
            pprint(response)
            pprint(stream)

        chunk = None

        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            if verbose:
                print("event::")
                pprint(event)
                print("chunk::")
                pprint(chunk)

                print("type::", chunk["type"])
            match chunk["type"]:
                case "message_start":
                    """
                    {'message': {'content': [],
                                'id': 'msg_017ns8c4urCpJrVpeNjatLyp',
                                'model': 'claude-3-haiku-48k-20240307',
                                'role': 'assistant',
                                'stop_reason': None,
                                'stop_sequence': None,
                                'type': 'message',
                                'usage': {'input_tokens': 33, 'output_tokens': 1}},
                    'type': 'message_start'}
                    """
                    emd["actual_model"] = chunk["message"]["model"]

                case "content_block_start":
                    """
                    {'content_block': {'text': '', 'type': 'text'},
                     'index': 0,
                    'type': 'content_block_start'}

                    """
                    pass

                case "content_block_delta":
                    """
                    {'chunk': {'bytes': b'{"type":"content_block_delta","index":0,"delta":{"type":'
                    b'"text_delta","text":","}}'}}
                    """
                    # emd["num_output_tokens"] += 1
                    text = chunk["delta"]["text"]
                    if "client_measured_time_to_first_token_s" not in emd:
                        emd["client_measured_time_to_first_token_s"] = (
                            time.time() - started
                        )

                    if print_fn:
                        print_fn(text)
                    emd["completion"] += text

                case "message_stop":
                    """
                    {'amazon-bedrock-invocationMetrics': {'firstByteLatency': 431,
                                                        'inputTokenCount': 33,
                                                        'invocationLatency': 2333,
                                                        'outputTokenCount': 200},
                    'type': 'message_stop'}
                    """
                    emd["metrics"] = chunk["amazon-bedrock-invocationMetrics"]

                case "content_block_stop":
                    """
                    {'chunk': {'bytes': b'{"type":"content_block_stop","index":0}'}}
                    """
                    pass

                case "message_delta":
                    """
                    {'delta': {'stop_reason': 'max_tokens', 'stop_sequence': None},
                    'type': 'message_delta',
                    'usage': {'output_tokens': 200}}
                    """
                    pass

                case _ as unknown_message_type:
                    raise RuntimeError(
                        f"Did not expect message of type: {unknown_message_type}."
                    )
        emd["client_measured_latency_s"] = time.time() - started
        return emd

    except ClientError as err:
        print(err)
        raise err


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-conf", type=int, default=0)
    parser.add_argument("--output-completions", type=int, default=0)
    parser.add_argument("--output-runs", type=int, default=0)
    parser.add_argument("--output-report", type=int, default=0)

    args, _ = parser.parse_known_args()

    conf = Config.from_yaml_file("config.yaml")
    if args.output_conf:
        pprint(conf)

    runs = []

    for scenario in conf.scenarios:
        if scenario.enabled:
            prompt_text = conf.prompts[scenario.prompt_id].text
            max_tokens = conf.prompts[scenario.prompt_id].max_tokens

            for model in scenario.models:
                model_id = conf.models[model].id

                for region in conf.models[model].regions:
                    runs.append(
                        {
                            "scenario": scenario.nick,
                            "prompt": prompt_text,
                            "max_tokens": max_tokens,
                            "model_id": model_id,
                            "region": region,
                        }
                    )
    if args.output_runs:
        pprint(runs)

    print("\nExecution starting.\n\n")

    report = ""

    for run in runs:
        results = generate(**run)
        metrics = results["metrics"]
        s = f'{run["scenario"]:>30s}'
        s += f'{results["actual_model"]:>30s} '
        s += f' {run["region"]:>15s} '
        s += f'{metrics["firstByteLatency"]/1000.:7.3f}s, '
        s += f'{metrics["invocationLatency"]/1000.:8.3f}s, '
        s += f'{results["client_measured_time_to_first_token_s"]:7.3f}s, '
        s += f'{results["client_measured_latency_s"]:7.3f}s, '
        s += f'{metrics["inputTokenCount"]:5d}, {metrics["outputTokenCount"]:5d} '

        report += s + "\n"
        print(s)

        if args.output_completions:
            print(results["completion"])

    if args.output_report:
        print("\nReport:\n")
        print(report)


if __name__ == "__main__":
    main()
