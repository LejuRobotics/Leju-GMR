#!/usr/bin/env python
# encoding=utf-8
"""
视频上传即文件下载工具脚本，用于将视频文件上传至千面动捕。
"""

import os
import sys
import argparse
import requests
import json
import time
from urllib.parse import quote

def upload_video(args):
    """
    执行上传逻辑 (适配新 COS 直传接口)
    """
    # 预处理 URL，确保以 / 结尾
    domain_url = args.domain_url
    if not domain_url.endswith("/"):
        domain_url += '/'

    base_url = f"{domain_url}business/"

    # 检查文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在 - {args.video_path}")
        sys.exit(1)

    # 第一步：获取 COS 上传凭证
    print(f"正在获取上传凭证...")
    cred_url = base_url + "uploadCosCredential"
    cred_payload = {
                    "companyKey": args.key,
                    "suffix": ".mp4"
                   }
    
    try:
        cred = requests.post(cred_url, json=cred_payload).json()
        cred_data = cred.get("data")
        if cred_data.get("status") != "200":
            print(f"获取凭证失败: {cred}")
            return None
            
        # 提取关键信息
        upload_url = cred_data["uploadUrl"]       # COS 上传地址
        cos_object_key = cred_data["cosObjectKey"] # COS 文件路径键
        print("凭证获取成功。")
        print(f"凭证获取成功  {cred_data.get("status")}")
        print(cred)

    except Exception as e:
        print(f"获取凭证请求异常: {str(e)}")
        return None

    # 第二步：直传文件到 COS
    print(f"正在上传视频文件: {args.video_path}")
    try:
        with open(args.video_path, "rb") as f:
            # 使用 put 方法直接上传二进制流
            requests.put(upload_url, data=f).raise_for_status()
        print("文件流上传完成。")

    except requests.exceptions.RequestException as e:
        print(f"COS 上传失败: {str(e)}")
        return None
    except Exception as e:
        print(f"文件读取错误: {str(e)}")
        return None

    # 确认上传 (通知业务服务器)
    print(f"正在确认上传信息...")
    confirm_url = base_url + "uploadCosConfirm"
    
    # 构建确认上传的 payload
    confirm_payload = {
        "companyKey": args.key,
        "cosObjectKey": cos_object_key,
        "videoName": os.path.basename(args.video_path).split('.')[0], # 去除后缀作为视频名
        "bonetype": args.bone_type,
        "capturetype": args.capture_type,
        "poseType": args.pose_type,
        "frameRate": args.frame_rate,
        "standPose": args.stand_pose,
        # "physicType": args.physic_type,
        "physicTimes": args.physic_times,
        "piercing" : args.piercing,
        "rollbackUrl": args.rollback_url,
    }

    try:
        confirm_resp = requests.post(confirm_url, json=confirm_payload)
        response_text = confirm_resp.text
        response_json = confirm_resp.json()
        
        print("-" * 30)

        # 尝试解析 videoId
        if response_json.get("data").get("status") == "200":
             video_id = response_json.get("data").get("videoId")
             print(f"上传流程结束，Video ID: {video_id}")
             print(response_text)
             return video_id
        else:
             print("上传确认失败或状态异常")
             print(response_text)
             return None

    except Exception as e:
        print(f"确认上传请求异常: {str(e)}")
        return None
    
def get_video_status(args, video_id):
    """
    轮询查询视频制作状态
    """
    # 预处理 URL
    domain_url = args.domain_url
    if not domain_url.endswith("/"):
        domain_url += '/'
    
    # 构建查询接口 URL (路径参数)
    # 注意：文档要求路径参数若含特殊字符需 URL 编码
    safe_video_id = quote(video_id, safe='')
    safe_company_key = quote(args.key, safe='')
    
    status_url = f"{domain_url}business/getStatus/{safe_video_id}/{safe_company_key}"
    
    print(f"--- 开始轮询视频状态 ---")
    print(f"Video ID: {video_id}")
    print(f"查询地址: {status_url}")
    print("-" * 30)

    # 轮询配置
    max_retries = 180         # 最大轮询次数
    retry_interval = 10       # 每次间隔秒数
    
    last_status = ""          # 用于记录上一次的状态，避免重复打印相同信息

    for attempt in range(max_retries):
        try:
            # 发送 POST 请求 (无请求体)
            response = requests.post(status_url)
            response.raise_for_status() # 检查 HTTP 错误
            
            resp_data = response.json()
            
            # 解析状态
            # 接口定义的 status 字段表示查询逻辑是否正常
            if resp_data.get("data").get("status") != "200":
                print(f"查询接口返回异常: {resp_data}")
                time.sleep(retry_interval)
                continue

            # 获取制作状态文案 (videoStatus)
            current_status = resp_data.get("data").get("videoStatus", "")
            message = resp_data.get("data").get("message", "")
            
            # 只有当状态发生变化时才打印
            if current_status not in ["待制作", "制作中", "制作完成"]:
                print(f"视频制作状态异常: {current_status}")
                return False
            
            if current_status != last_status:
                print(f"[{attempt + 1}/{max_retries}] 当前状态: {current_status}")
                print(f"   详细信息: {message}")
                last_status = current_status
            if current_status == "制作完成":
                print(f"上传参数快照: {resp_data.get("data").get("data", "")}")
                return True

        except Exception as e:
            print(f"轮询请求异常: {str(e)}")
        
        # 等待下一次轮询
        if attempt < max_retries - 1:
            time.sleep(retry_interval)
        else:
            print("-" * 30)
            print(f"已达到最大轮询次数 ({max_retries})，视频可能仍在制作中。")
            print(f"最后一次已知状态: {last_status}")
            return False

    return False


def download_file(args, video_id):
    """
    执行下载
    """
    # 预处理 URL
    domain_url = args.domain_url
    if not domain_url.endswith("/"):
        domain_url += '/'
    
    # 新的 API 端点
    url = f"{domain_url}business/downloadCosCredential"
    
    # 请求参数
    payload = {
        "companyKey": args.key,
        "videoId": video_id,
        "videoSign": args.video_sign,
    }

    print(f"\n正在获取下载凭证...")
    print(f"请求接口: {url}")

    try:
        # 第一步：调用接口获取下载链接
        response = requests.post(url, json=payload)
        response.raise_for_status() # 检查 HTTP 状态码
        
        resp_data = response.json()
        
        # 检查业务状态码
        if resp_data.get("data").get("status") != "200":
            print(f"获取下载凭证失败: {resp_data.get("data").get('message')}")
            return

        # 第二步：解析返回数据
        archive_name = resp_data.get("data").get("archiveName")
        files_list = resp_data.get("data").get("files", [])
        
        if not files_list:
            print("未找到文件列表，可能任务尚未完成或数据异常。")
            return

        # 确保输出目录存在
        output_dir_path = args.output_dir_path
        os.makedirs(output_dir_path, exist_ok=True)

        # 第三步：查找并下载 BVH 文件
        target_file = None
        target_url = None
        
        # 在文件列表中寻找 .bvh 文件
        for file_info in files_list:
            file_name = file_info["fileName"]
            if file_name.lower().endswith('.bvh'): # .lower() 防止大小写问题
                target_file = file_name
                target_url = file_info["downloadUrl"]
                break # 找到第一个 bvh 文件即停止
        
        if not target_file:
            print("未找到 BVH 文件，请检查。")
            return

        # 确保输出目录存在
        output_dir_path = args.output_dir_path
        os.makedirs(output_dir_path, exist_ok=True)
        
        output_file_path = os.path.join(output_dir_path, target_file)
        
        print(f"正在下载 BVH 文件: {target_file} ...")
        
        # 使用 stream=True 进行流式下载
        with requests.get(target_url, stream=True) as r:
            r.raise_for_status()
            with open(output_file_path, 'wb') as f:
                # 分块写入
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        print(f"成功！BVH 文件已保存至: {output_file_path}")

    except requests.exceptions.RequestException as e:
        print(f"网络请求异常: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"下载过程发生错误: {str(e)}")
        sys.exit(1)


def main():
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(
        description="通用视频上传脚本 - 支持自定义 API 参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数上传
  python qmai_video_uploader.py -f data/video.mp4
  # 指定所有参数
  python qmai_video_uploader.py -f video.mp4 -k "your Key" -u "http://example.com" -c 1 -b 0
        """
    )

    # 核心参数
    parser.add_argument('-f', '--file', dest='video_path', required=True, help='Path of video file(required)')
    # 认证与连接
    parser.add_argument('-u', '--url', dest='domain_url', default="https://www.qmai.vip", help='API Domain Address (default: https://www.qmai.vip)')
    parser.add_argument('-k', '--key', dest='key', required=True, help='Company Key(required)')
    # 上传参数
    parser.add_argument('-c', '--capture', dest='capture_type', default="0", help='Capture Type, multiple values separated by commas, such as "0,2,3". 0:Whole body (default), 1: Half body, 2: Hand catching, 3: Face catching, 5: Automatic judgment (full body/half body)')
    parser.add_argument('-b', '--bone', dest='bone_type', default= 15, help='Bone Type, refer to the skeleton type table')
    parser.add_argument('-p', '--pose', dest='pose_type', default= 1, help='First frame pose: 1 (TPose), 2 (APose), 3 (original pose)(default: 3)')
    parser.add_argument('-r', '--frameRate', dest='frame_rate', default= 30, help='Output frame rate (Billing-related): 24/30, 60, 120')
    parser.add_argument('-s', '--standPose', dest='stand_pose', action="store_true", default=False, help='Move in place: (default: False)')
    # parser.add_argument('--physicType', dest='physic_type', default="2", help='Physical optimization type 2(2.0) (default: 2 open)')
    parser.add_argument('--physicTimes', dest='physic_times', default= 6, help='Only valid when physicalType=2, optional 1-6 (6 represents 10 times) (default: 6)')
    parser.add_argument('--piercing', dest='piercing', default= "0", help='Anti-mold protection, optional 0, 1 (default 0 is off)')
    
    #下载参数
    parser.add_argument('-o', '--output_dir_path', dest='output_dir_path', default="bvh_data", help='output dir path')
    parser.add_argument('--videoSign', dest='video_sign', default= 1, help='When sending option 1, the list may include processed video, audio, etc. (if they exist); otherwise, the default file list will be used.')
    parser.add_argument('--isDownload', dest='is_download', action="store_true", default=True, help='Whether download file (default: True)')

    # 回调配置
    parser.add_argument('--rollback', dest='rollback_url', default="http://192.168.141.68:80", help='Callback notification URL')

    args = parser.parse_args()
    
    # 执行上传
    video_id = upload_video(args)

    #查询状态
    download_tag = False
    if video_id != None:
        download_tag = get_video_status(args, video_id)

    #执行下载
    if args.is_download and download_tag:
        download_file(args, video_id)

if __name__ == "__main__":
    main()
