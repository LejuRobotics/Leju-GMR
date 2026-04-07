#!/usr/bin/env python
# encoding=utf-8
"""
视频上传即文件下载工具脚本，用于将视频文件上传至千面动捕。
"""

import os
import sys
import argparse
import requests
import base64
import json
import traceback
import time


def upload_video(args):
    """
    执行上传逻辑
    """
    # 1. 预处理 URL，确保以 / 结尾
    domain_url = args.domain_url
    if not domain_url.endswith("/"):
        domain_url += '/'

    # 2. 检查文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在 - {args.video_path}")
        sys.exit(1)

    # 3. 准备 API 端点
    url = f"{domain_url}business/upload/"
    
    # 4. 准备数据载荷
    # 使用字典构建 data，方便处理
    data = {
        "capturetype": args.capture_type,
        "bonetype": args.bone_type,
        "companyKey": args.key,
        "rollbackUrl": args.rollback_url,
        "pose_type": args.pose_type,
        "frameRate": args.frame_rate,
        "standPose": args.stand_pose,
        "physicType": args.physic_type,
        "physicTimes": args.physic_times
    }

    print(f"正在连接: {url}")
    print(f"准备上传: {args.video_path}")
    print("开始上传... (请稍候)")

    try:
        # 打开文件并上传
        # 原生 requests 会自动处理 multipart/form-data
        with open(args.video_path, 'rb') as video_file:
            files = {'videoFile': (os.path.basename(args.video_path), video_file, 'video/mp4')}
            response = requests.post(url, data=data, files=files, timeout=600)

        # 5. 处理响应
        print("-" * 30)
        if response.status_code == 200:
            print("上传成功 (200 OK)")
            print("响应内容:")
            print(response.text)
            response_json = response.json()
            video_id = response_json.get("data", {}).get("videoId")
            print(video_id)
            return video_id
        else:
            print(f"上传失败: 状态码 {response.status_code}")
            print("错误详情:")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"网络请求异常: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"文件读取或其他错误: {str(e)}")
        sys.exit(1)
    return 

def download_file(args, video_id):

    # Configuration
    domain_url = args.domain_url

    # Ensure URL ends with "/"
    if not domain_url.endswith("/"):
        domain_url += '/'

    # API endpoint URL
    url = f"{domain_url}business/download/"
    # Request data
    data = {
        "videoId": video_id,
        "companyKey": args.key
    }

    print(f"\n开始轮询下载 {data}...")
    print(f"下载接口: {url}")

    # 轮询逻辑配置
    max_retries = 20 # 最大重试次数
    retry_interval = 60 # 每次间隔秒数

    for attempt in range(max_retries):
        print(f"轮询尝试 {attempt + 1}/{max_retries} ...")
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                response_data = response.json()
                try:
                    # code 0 且 status 200 代表成功
                    if (response_data.get("code") == 0 and 
                        response_data.get("data", {}).get("status") == "200"):
                        data = response_data["data"]
                        file_name = data["fileName"]
                        output_dir_path = args.output_dir_path
                        os.makedirs(output_dir_path, exist_ok=True)
                        output_file_path = os.path.join(output_dir_path, file_name)
                        file_bytes = base64.b64decode(data["file"])
                        with open(output_file_path, "wb") as out:
                            out.write(file_bytes)
                        print(f"\n成功！文件已下载至: {output_file_path}")
                        return # 下载成功，退出函数
                    else:
                        # 任务未完成，打印状态供调试
                        status = response_data.get("data", {}).get("status", "Unknown")
                        print(f"任务未完成，当前状态: {status}。{retry_interval}秒后重试...")
                except Exception as e:
                    print(f"响应解析异常 (可能任务还在处理中): {e}")
            else:
                print(f"下载接口请求失败: {response.status_code}")
        except Exception as e:
            print(f"请求异常: {e}")
        # 如果没有成功，等待一段时间再重试
        if attempt < max_retries - 1: # 最后一次尝试后不再等待
            time.sleep(retry_interval)
        else:

            print(f"\n错误: 已达到最大重试次数 ({max_retries})，任务可能仍在处理或已失败")
            print("\n建议去网页端或通过其他接口查询任务状态")


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
    parser.add_argument('-k', '--key', dest='key', required=True, help='Company Key / API Key(required)')
    # 上传参数
    parser.add_argument('-c', '--capture', dest='capture_type', default="0", help='Capture Type, multiple values separated by commas, such as "0,2,3". 0:Whole body (default), 1: Half body, 2: Hand catching, 3: Face catching, 5: Automatic judgment (full body/half body)')
    parser.add_argument('-b', '--bone', dest='bone_type', default="15", help='Bone Type, refer to the skeleton type table')
    parser.add_argument('-p', '--pose', dest='pose_type', default="1", help='First frame pose: 1 (TPose), 2 (APose), 3 (original pose)(default: 1)')
    parser.add_argument('-r', '--frameRate', dest='frame_rate', default="30", help='Output frame rate (Billing-related): 24/30, 60, 120')
    parser.add_argument('-s', '--standPose', dest='stand_pose', action="store_true", default=False, help='Move in place: (default: False)')
    parser.add_argument('--physicType', dest='physic_type', default="1", help='Physical optimization type (Billing-related): 1(1.0), 2(2.0) (default: 1)')
    parser.add_argument('--physicTimes', dest='physic_times', default="6", help='Only valid when physicalType=2, optional 1-6 (6 represents 10 times) (default: 6)')
    
    #下载参数
    parser.add_argument('-o', '--output_dir_path', dest='output_dir_path', default="output/bvh", help='output dir path')
    parser.add_argument('--isDownload', dest='is_download', action="store_true", default=True, help='Whether download file (default: True)')

    # 回调配置
    parser.add_argument('--rollback', dest='rollback_url', default="http://192.168.141.68:80", help='Callback notification URL')

    args = parser.parse_args()
    
    # 执行上传
    video_id = upload_video(args)

    #执行下载
    if args.is_download:
        download_file(args, video_id)

if __name__ == "__main__":       
    main()