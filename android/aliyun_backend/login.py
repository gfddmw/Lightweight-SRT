# -*- coding: utf-8 -*-
import json
import logging
import base64
from tablestore import *

# --- 配置区 ---
OTS_ENDPOINT = 'https://srt.cn-hangzhou.ots.aliyuncs.com'
OTS_INSTANCE = 'srt'
MAIN_TABLE = 'user_profiles'  # 主表名
INDEX_NAME = 'email'          # 二级索引名

def handler(event, context):
    logger = logging.getLogger()
    try:
        # 1. 解析 API 网关触发的 event 数据
        if isinstance(event, dict):
            payload = event
        else:
            payload = json.loads(event.decode('utf-8'))

        body_data = payload.get('body', '')
        if payload.get('isBase64Encoded', False):
            body_data = base64.b64decode(body_data).decode('utf-8')

        params = json.loads(body_data) if isinstance(body_data, str) else body_data

        username_input = params.get('username')
        password_input = params.get('password')

        if not username_input or not password_input:
            return {'statusCode': 400, 'body': json.dumps({'error': '用户名或密码不能为空'})}

        # 2. 初始化 OTS 客户端
        creds = context.credentials
        client = OTSClient(OTS_ENDPOINT, creds.access_key_id, creds.access_key_secret,
                           OTS_INSTANCE, sts_token=creds.security_token)

        target_row = None

        # 3. 执行查询逻辑
        if "@" in str(username_input):
            # --- 解决报错的核心：使用 get_range 查询索引表 ---
            # 索引表主键结构为：[email (索引键), username (原表主键)]
            # 因为我们不知道 username，所以查询 (email, 最小值) 到 (email, 最大值) 之间的行
            logger.info(f"Using GSI Range Query for email: {username_input}")

            inclusive_start_pk = [('email', str(username_input)), ('username', INF_MIN)]
            exclusive_end_pk = [('email', str(username_input)), ('username', INF_MAX)]

            # 执行范围查询
            consumed, next_start_pk, row_list, request_id = client.get_range(
                INDEX_NAME,
                Direction.FORWARD,
                inclusive_start_pk,
                exclusive_end_pk,
                columns_to_get=['password'],
                limit=1
            )
            if row_list:
                target_row = row_list[0]
        else:
            # 使用主表主键点查询
            logger.info(f"Using Main Table Query for username: {username_input}")
            primary_key = [('username', str(username_input))]
            _, target_row, _ = client.get_row(MAIN_TABLE, primary_key, columns_to_get=['password'])

        # 4. 结果校验
        if target_row:
            # 从 attribute_columns 中提取密码
            db_pwd = None
            for col in target_row.attribute_columns:
                if col[0] == 'password':
                    db_pwd = col[1]
                    break

            if db_pwd and str(db_pwd) == str(password_input):
                # 提取 username 和 email
                res_username = ""
                res_email = ""
                
                for pk in target_row.primary_key:
                    if pk[0] == 'username':
                        res_username = pk[1]
                    elif pk[0] == 'email':
                        res_email = pk[1]
                
                # 容错处理
                if not res_email and "@" in str(username_input):
                    res_email = username_input
                if not res_username and "@" not in str(username_input):
                    res_username = username_input

                # 构造符合 Android AuthData 结构的返回
                auth_data = {
                    "user": {
                        "id": 0,
                        "username": res_username,
                        "email": res_email,
                        "password": "" 
                    },
                    "access_token": "mock_access_token_" + res_username,
                    "refresh_token": "mock_refresh_token_" + res_username
                }

                return {
                    'statusCode': 200,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps(auth_data)
                }
            return {'statusCode': 401, 'body': json.dumps({'error': '密码错误'})}

        return {'statusCode': 404, 'body': json.dumps({'error': '用户不存在'})}

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error', 'detail': str(e)})
        }
