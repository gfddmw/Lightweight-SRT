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
    """
    构建符合阿里云 API 网关标准的响应结构
    """
    return {
        "isBase64Encoded": False,
        "statusCode": int(status_code),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body_dict)
    }

def handler(event, context):
    logger = logging.getLogger()
    try:
        # 1. 解析参数
        if isinstance(event, dict):
            payload = event
        else:
            payload = json.loads(event.decode('utf-8'))

        # 尝试从所有可能的网关字段中获取 username
        # 1. queryParameters (常见)
        # 2. queryStringParameters (阿里云某些触发器使用)
        # 3. payload 根节点 (直接调用或透传模式)
        # 4. pathParameters (路径参数模式)

        q_params = payload.get('queryParameters') or payload.get('queryStringParameters') or {}
        p_params = payload.get('pathParameters') or {}

        username = q_params.get('username') or payload.get('username') or p_params.get('username')

        if not username:
            # 记录一下 payload 结构，方便在日志里排查
            logger.error(f"Missing username. Payload keys: {list(payload.keys())}")
            return build_response(400, {'error': 'username required', 'received_keys': list(payload.keys())})
        # 2. 初始化 OTS 客户端
        creds = context.credentials
        if not creds:
            return build_response(500, {'error': 'FC context credentials missing'})

        client = OTSClient(OTS_ENDPOINT, creds.access_key_id, creds.access_key_secret,
                           OTS_INSTANCE, sts_token=creds.security_token)

        # 3. 执行查询
        primary_key = [('username', str(username))]
        columns_to_get = ['nickname', 'description', 'avatar_url']

        consumed, row, next_token = client.get_row(MAIN_TABLE, primary_key, columns_to_get)

        if row:
            attrs = {col[0]: col[1] for col in row.attribute_columns}
            return build_response(200, {
                'status': 'success',
                'profile': {
                    'nickname': attrs.get('nickname', str(username)),
                    'description': attrs.get('description', ''),
                    'avatarUrl': attrs.get('avatar_url', '')
                }
            })
        else:
            return build_response(404, {'error': 'User not found'})

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return build_response(500, {'error': 'Internal Server Error', 'detail': str(e)})
