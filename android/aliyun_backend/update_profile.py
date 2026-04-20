# -*- coding: utf-8 -*-
import json
import logging
import base64
from tablestore import *

# --- 配置区 ---
OTS_ENDPOINT = 'https://srt.cn-hangzhou.ots.aliyuncs.com'
OTS_INSTANCE = 'srt'
MAIN_TABLE = 'user_profiles'

def build_response(status_code, body_dict):
    return {
        "statusCode": int(status_code),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body_dict)
    }

def handler(event, context):
    logger = logging.getLogger()
    # 增加日志输出，确保一旦进入函数就能看到
    logger.info(f"Update Profile Triggered. Event: {str(event)[:500]}")
    
    try:
        if isinstance(event, dict):
            payload = event
        else:
            payload = json.loads(event.decode('utf-8'))

        # 1. 尝试从各种地方抓取参数
        # 尝试 Query 参数 (网关可能强制要求的)
        q_params = payload.get('queryParameters') or payload.get('queryStringParameters') or {}
        
        # 尝试 Body (POST 正常存放处)
        body_raw = payload.get('body', '')
        if payload.get('isBase64Encoded', False):
            try:
                body_raw = base64.b64decode(body_raw).decode('utf-8')
            except: pass
            
        b_params = {}
        if body_raw:
            try:
                b_params = json.loads(body_raw)
            except: pass

        # 综合获取所有可能的参数来源
        # 优先级：Body > Query > Root
        params = {}
        params.update(q_params)
        if isinstance(b_params, dict): params.update(b_params)
        if isinstance(payload, dict):
            for k, v in payload.items():
                if k not in ['body', 'queryParameters', 'queryStringParameters']:
                    params[k] = v

        username = params.get('username')
        nickname = params.get('nickname')
        description = params.get('description')
        avatar_url = params.get('avatarUrl') or params.get('avatar_url')

        if not username:
            return build_response(400, {'error': 'username missing', 'debug_keys': list(params.keys())})

        # 2. 初始化 OTS
        creds = context.credentials
        client = OTSClient(OTS_ENDPOINT, creds.access_key_id, creds.access_key_secret,
                           OTS_INSTANCE, sts_token=creds.security_token)

        # 3. 执行更新
        primary_key = [('username', str(username))]
        attributes = []
        if nickname is not None: attributes.append(('nickname', str(nickname)))
        if description is not None: attributes.append(('description', str(description)))
        if avatar_url: attributes.append(('avatar_url', str(avatar_url)))

        if attributes:
            client.update_row(MAIN_TABLE, Row(primary_key, {'PUT': attributes}), None)
            return build_response(200, {'status': 'success', 'message': 'Profile updated'})
        else:
            return build_response(200, {'status': 'success', 'message': 'nothing to update'})

    except Exception as e:
        logger.error(f"Internal Error: {str(e)}")
        return build_response(500, {'error': str(e)})
