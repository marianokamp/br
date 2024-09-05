#!/usr/bin/env python

import logging
import re
from re import template
import time
import json
import argparse
import datetime
import uuid
import random
from pprint import pprint
import boto3
from botocore.exceptions import ClientError

from config import Config

logger = logging.getLogger(__name__)

bedrock_admin = boto3.client(service_name="bedrock", region_name="us-east-1")


class ModelProviderAdapter:
    def __init__(self, model_id):
        self.model_id = model_id

    def adapt_prompt(self, prompt) -> str:
        raise RuntimeError("Not implemented")

    def adapt_body(self, prompt, max_tokens, temperature=0.0, top_p=1.0) -> str:
        raise RuntimeError("Not implemented")

    def handle_chunk(self, chunk, emd, print_fn) -> None:
        raise RuntimeError("Not implemented")

    @classmethod
    def instantiate(cls, model_id):
        if "anthropic" in model_id:
            return AnthropicModelAdapter(model_id)
        elif "mistral" in model_id:
            return MistralModelAdapter(model_id)
        elif "llama" in model_id:
            return LlamaModelAdapter(model_id)
        raise RuntimeError(f"No model provider adapter for {model_id}.")


class AnthropicModelAdapter(ModelProviderAdapter):
    def adapt_body(self, prompt, max_tokens, temperature=0.0, top_p=1.0):
        return json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    def adapt_prompt(self, prompt):
        return prompt

    def handle_chunk(self, chunk, emd, print_fn):
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
                        time.time() - emd["started"]
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


class LlamaModelAdapter(ModelProviderAdapter):
    def adapt_body(self, prompt, max_tokens, temperature=0.0, top_p=1.0):
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return body

    def adapt_prompt(self, prompt):
        adapted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return adapted_prompt

    def handle_chunk(self, chunk, emd, print_fn):
        if "actual_model" not in emd:
            emd["actual_model"] = self.model_id
        for output in chunk["generation"]:
            text = output
            if text:
                if "client_measured_time_to_first_token_s" not in emd:
                    emd["client_measured_time_to_first_token_s"] = (
                        time.time() - emd["started"]
                    )

                emd["completion"] += text
                if print_fn:
                    print_fn(text)

        if "amazon-bedrock-invocationMetrics" in chunk:
            emd["metrics"] = chunk["amazon-bedrock-invocationMetrics"]


class MistralModelAdapter(ModelProviderAdapter):
    def adapt_body(self, prompt, max_tokens, temperature=0.0, top_p=1.0):
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return body

    def adapt_prompt(self, prompt):
        return f"<s>[INST]{prompt}[/INST]"
        # FIXME: No trailing </s>. That would be weird.

    def handle_chunk(self, chunk, emd, print_fn):
        if "actual_model" not in emd:
            emd["actual_model"] = self.model_id
        for output in chunk["outputs"]:
            text = output["text"]
            if text:
                if "client_measured_time_to_first_token_s" not in emd:
                    emd["client_measured_time_to_first_token_s"] = (
                        time.time() - emd["started"]
                    )

                emd["completion"] += text
                if print_fn:
                    print_fn(text)

        if "amazon-bedrock-invocationMetrics" in chunk:
            emd["metrics"] = chunk["amazon-bedrock-invocationMetrics"]


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
    model_adapter = ModelProviderAdapter.instantiate(model_id)
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)

    emd = dict(
        region=region,
        prompt=prompt,
        completion="",
        model_id=model_id,
        executed_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )  # Execution metadata

    adapted_prompt = model_adapter.adapt_prompt(prompt)

    # FIXME: Merge with adapt_prompt?
    body = model_adapter.adapt_body(
        adapted_prompt, max_tokens, temperature=0.0, top_p=1.0
    )
    emd["started"] = time.time()
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

            model_adapter.handle_chunk(chunk, emd, print_fn)

        emd["client_measured_latency_s"] = time.time() - emd["started"]
        return emd

    except ClientError as err:
        print(f"Error when accessing {model_id} in {region}: {err}")
        raise err


def run_and_report(conf, run):
    result = generate(**run)

    d = {**result, **run}
    s = json.dumps(d)

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=conf.locations.run_report_bucket,
        Key=conf.locations.run_report_prefix
        + "/"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + "_"
        + str(uuid.uuid4())[:8]
        + ".json",
        Body=s.encode("utf-8"),
    )
    return result


def create_runs(conf):
    """
    Creates all runs for scenarios that are enabled.
    """

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
    return runs


def report_run(run, results):
    metrics = results["metrics"]
    s = f'{run["scenario"]:>30s}'
    s += f' {results["model_id"]:>40s} '
    s += f'{results["actual_model"]:>35s} '
    s += f' {run["region"]:>15s} '
    s += f'{metrics["firstByteLatency"]/1000.:7.3f}s, '
    s += f'{metrics["invocationLatency"]/1000.:8.3f}s, '
    s += f'{results["client_measured_time_to_first_token_s"]:7.3f}s, '
    s += f'{results["client_measured_latency_s"]:7.3f}s, '
    s += f'{metrics["inputTokenCount"]:5d}, {metrics["outputTokenCount"]:5d} '

    print(s)
    return s


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-conf", type=int, default=0)
    parser.add_argument("--output-completions", type=int, default=0)
    parser.add_argument("--output-runs", type=int, default=0)
    parser.add_argument("--output-report", type=int, default=0)
    parser.add_argument("--run-local-reports", type=int, default=0)
    parser.add_argument("--run-reports", type=int, default=0)
    parser.add_argument("--list-models", type=int, default=0)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)

    args, _ = parser.parse_known_args()

    if args.list_models:
        list_models()

    conf = Config.from_yaml_file("config.yaml")
    if args.output_conf:
        pprint(conf)

    runs = create_runs(conf)
    if args.filter:
        before = len(runs)
        runs = [r for r in runs if re.search(args.filter, str(r))]
        print(f"Using filtering, from {before} to {len(runs)} elements.")

    if args.output_runs:
        pprint(runs)

    if args.run_local_reports:
        print("\nExecution starting.\n")

        report = ""
        print(
            "Scenario, req model id, actual model id, region, \n"
            "reported first byte latency, reported invocation latency, client measured time to first token, client measured latency,\n"
            "input token count, output token count"
        )

        for run in runs:
            results = generate(**run, verbose=args.verbose)
            s = report_run(run, results)
            report += s + "\n"

            if args.output_completions:
                print(results["completion"])

        if args.output_report:
            print("\nReport:\n")
            print(report)

    if args.run_reports:
        while True:
            print(datetime.datetime.now(), end="\t")
            run = random.choice(runs)
            try:
                result = run_and_report(conf, run)
                report_run(run, result)
            except ClientError as err:
                print("err", err)
                print(run["scenario"], run["model_id"], run["region"])

            wait_s = random.randint(5 * 60, 20 * 60)
            time.sleep(wait_s)


if __name__ == "__main__":
    main()
