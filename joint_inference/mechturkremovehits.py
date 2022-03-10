import csv
import os
import pickle
from datetime import datetime
from random import shuffle

import boto3

region_name = 'us-east-1'
aws_access_key_id = ''
aws_secret_access_key = ''

import pandas as pd

hits = []
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Push mechanical turk questions.')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start question index (inclusive)')
    parser.add_argument('--end_index', type=int, default=30,
                        help='End question index (exclusive)')
    parser.add_argument('--endpoint', default="dev",
                        help='Choose dev or prod')
    args = parser.parse_args()

    start_question_index = args.start_index
    end_question_index = args.end_index

    skip_first = False
    limit = 1

    endpoints = {
        "dev": 'https://mturk-requester-sandbox.us-east-1.amazonaws.com',
        "prod": 'https://mturk-requester.us-east-1.amazonaws.com'
    }
    endpoint_url = endpoints[args.endpoint]

    # Uncomment this line to use in production
    # endpoint_url =

    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # This will return $10,000.00 in the MTurk Developer Sandbox
    print(mturk.get_account_balance()['AvailableBalance'])

    for item in mturk.list_hits(MaxResults=100)['HITs']:
        hit_id = item['HITId']
        print('HITId:', hit_id)

        # Get HIT status
        status = mturk.get_hit(HITId=hit_id)['HIT']['HITStatus']
        print('HITStatus:', status)

        # If HIT is active then set it to expire immediately
        if status == 'Assignable':
            response = mturk.update_expiration_for_hit(
                HITId=hit_id,
                ExpireAt=datetime(2015, 1, 1)
            )

            # Delete the HIT
        try:
            mturk.delete_hit(HITId=hit_id)
        except:
            print('Not deleted')
        else:
            print('Deleted')
