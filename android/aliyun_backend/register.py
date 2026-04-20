# -*- coding: utf-8 -*-
import json
import logging
from tablestore import *

# 已根据您的截图自动填入真实信息
OTS_ENDPOINT = 'https://srt.cn-hangzhou.ots.aliyuncs.com'
OTS_INSTANCE = 'srt'

def handler(event, context):
    logger = logging.getLogger()
    try:
        # 解析参数 (处理网关包裹)
        data = json.loads(event.decode('utf-8'))
        body = json.loads(data.get('body', '{}')) if 'body' in data else data

        username = body.get('username')
        password = body.get('password')
        email = body.get('email', '')

        if not username or not password:
            return {'statusCode': 400, 'body': json.dumps({'error': '必填项缺失'})}

        # 写入 Tablestore
        creds = context.credentials
        client = OTSClient(OTS_ENDPOINT, creds.access_key_id, creds.access_key_secret, OTS_INSTANCE, sts_token=creds.security_token)

        primary_key = [('username', str(username))]
        attribute_columns = [
            ('password', str(password)),
            ('email', str(email)),
            ('nickname', str(username)),
            ('created_at', '2026-04-15')
        ]

        # Condition: EXPECT_NOT_EXIST 确保用户名不重复
        client.put_row('user_profiles', Row(primary_key, attribute_columns), Condition(RowExistenceExpectation.EXPECT_NOT_EXIST))

        # 构造符合 Android AuthData 结构的返回
        auth_data = {
            "user": {
                "id": 0,
                "username": str(username),
                "email": str(email),
                "password": "" 
            },
            "access_token": "mock_access_token_" + str(username),
            "refresh_token": "mock_refresh_token_" + str(username)
        }

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(auth_data)
        }

    except OTSClientError as e:
        if "Condition check failed" in str(e):
            return {'statusCode': 409, 'body': json.dumps({'error': '用户名已存在'})}
        return {'statusCode': 500, 'body': json.dumps({'error': '数据库写入失败', 'detail': str(e)})}

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
