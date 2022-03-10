import argparse
import csv
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from random import shuffle

import boto3

region_name = 'us-east-1'
aws_access_key_id = ''
aws_secret_access_key = ''


hits = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Push mechanical turk questions.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='File that contains the assertions we want to test')
    parser.add_argument('--hint',action='store_true',default=False,
                        help='Whether to use or not the generations with hints')
    parser.add_argument('--endpoint', default="dev",
                        help='Choose dev or prod')
    args = parser.parse_args()

    if args.endpoint not in ["dev","prod"]:
        raise Exception("Endpoint should be dev or prod")
    if args.endpoint == "dev":
        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    elif args.endpoint == "prod":
        # Uncomment this line to use in production
        endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    print("Endpoint:",endpoint_url)
    client = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # This will return $10,000.00 in the MTurk Developer Sandbox
    print(client.get_account_balance()['AvailableBalance'])

    skip_first = False
    count = 0
    for idx, line in enumerate(csv.reader(open(args.input_file))):
        if not skip_first:
            skip_first = True
            continue
        try:
            question = open('mechturkfiles/baseform.html', mode='r').read()
            # generation,hint,model_file,test_file,input,gold
            statement = line[0].replace("<general>","").replace("<specific>","").replace("<subject>","").replace("<object>",",").replace("<relation>",",").replace("</relation>","").strip()
            if line[1]=="no_hint" and args.hint:
                print("Skipping")
                continue
            context = line[4]
            if line[1]=="hint":
                if not args.hint:
                    print("Skipping")
                    continue
                context = context.split("(")[0].strip()
            print("Not skipping",line[1])
            context = context.replace("<s>","").replace("</s>","").replace("<story>","")
            sentence = context.split("<sentence>")[1]
            context = context.split("<sentence>")[0].strip()
            orig_ctx = deepcopy(context)
            context = context.replace(sentence.strip(), "<i>"+sentence.strip()+"</i>")
            if orig_ctx == context:
                print("as/da/!")
                print("asd")
            if len(line)==7:
                gold = line[-2].replace("<general>","").replace("<specific>","").replace("<subject>","").replace("<object>",",").replace("<relation>",",").replace("</relation>","")

            else:
                gold = line[-1].replace("<general>","").replace("<specific>","").replace("<subject>","").replace("<object>",",").replace("<relation>",",").replace("</relation>","")
            question = question.replace("<1>",statement)
            question = question.replace("<2>",context)
            question = question.replace("<3>",statement)
            question = question.replace("<4>",gold)
            if len(line)==7:
                question = question.replace("<5>",'<details>'+args.input_file+";hint="+str(args.hint)+";"+"score="+str(line[-1])+'</details>')
            else:
                question = question.replace("<5>",'<details>'+args.input_file+";hint="+str(args.hint)+'</details>')
            worker_requirements = [{
                'QualificationTypeId': '000000000000000000L0',
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [80],
                'RequiredToPreview': True,
            }]
            new_hit = client.create_hit(
                Title = 'Review Commonsense Statements',
                Description = 'Help review some commonsense statements in a given context' ,
                Keywords = 'review, survey, question, research',
                Reward = '0.1',
                MaxAssignments = 2,
                LifetimeInSeconds = 172800,
                AssignmentDurationInSeconds = 660,
                Question = question,
                QualificationRequirements=worker_requirements
            )
            print("A new HIT has been created. You can preview it here:")
            print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
            print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
            hits.append((idx,new_hit['HIT']['HITGroupId'], new_hit['HIT']['HITId'],"https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'],str(datetime.now())))
            pickle.dump(hits, open("hits.txt", "wb"))
            count+=1
            print(count)
            # break
        except Exception as e:
            print(e)
    print("Push count",count)
    pickle.dump(hits,open("hits.txt","wb"))