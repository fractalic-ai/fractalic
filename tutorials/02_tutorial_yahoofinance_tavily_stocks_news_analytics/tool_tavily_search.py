#!/usr/bin/env python3

import argparse
import os
import requests
import json
import sys


TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if not TAVILY_API_KEY:
    print("Error: TAVILY_API_KEY environment variable is not set.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Interact with the Tavily Search and Extract API.')

    parser.add_argument('--task', required=True, choices=['search', 'extract'], help='API task to perform.')
    parser.add_argument('--query', required=True, help='Search query or comma-separated URLs for extraction.')
    parser.add_argument('--search_depth', choices=['basic', 'advanced'], default='basic', help='Depth of the search.')
    parser.add_argument('--topic', choices=['general', 'news'], default='general', help='Search topic.')
    parser.add_argument('--days', type=int, default=3, help='Days back for news topic.')
    parser.add_argument('--max_results', type=int, default=5, help='Maximum number of results to return.')
    parser.add_argument('--include_images', action='store_true', help='Include images in search results.')
    parser.add_argument('--include_image_descriptions', action='store_true', help='Include image descriptions (requires --include_images).')
    parser.add_argument('--include_answer', action='store_true', help='Include a generated answer.')
    parser.add_argument('--include_raw_content', action='store_true', help='Include raw content of results.')
    parser.add_argument('--include_domains', help='Comma-separated list of domains to include.')
    parser.add_argument('--exclude_domains', help='Comma-separated list of domains to exclude.')

    return parser.parse_args()

def build_payload(args):
    if args.task == 'search':
        payload = {
            'query': args.query,
            'search_depth': args.search_depth,
            'topic': args.topic,
            'days': args.days,
            'max_results': args.max_results,
            'include_images': args.include_images,
            'include_image_descriptions': args.include_image_descriptions,
            'include_answer': args.include_answer,
            'include_raw_content': args.include_raw_content,
            'include_domains': args.include_domains.split(',') if args.include_domains else [],
            'exclude_domains': args.exclude_domains.split(',') if args.exclude_domains else []
        }
    else:  # extract task
        payload = {
            'urls': [url.strip() for url in args.query.split(',')]
        }

    return payload

def call_api(endpoint, payload):
    headers = {
        'Authorization': f'Bearer {TAVILY_API_KEY}',
        'Content-Type': 'application/json'
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.ok:
        print('Success:')
        print(json.dumps(response.json(), indent=2))
    else:
        print(f'Error: Received HTTP status {response.status_code}')
        print('Response:')
        print(response.text)
        sys.exit(1)

def main():
    args = parse_args()

    if args.include_image_descriptions and not args.include_images:
        print('Error: --include_image_descriptions requires --include_images.')
        sys.exit(1)

    endpoint = 'https://api.tavily.com/search' if args.task == 'search' else 'https://api.tavily.com/extract'

    payload = build_payload(args)

    call_api(endpoint, payload)

if __name__ == '__main__':
    args = parse_args()
    main()