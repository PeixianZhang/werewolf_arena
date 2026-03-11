#!/usr/bin/env python3
"""快速查看所有游戏第一夜死亡情况"""
import json
from pathlib import Path

def check_first_night_deaths():
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    logs_dir = project_root / "logs"
    
    if not logs_dir.exists():
        print(f"错误: 找不到 logs 目录: {logs_dir}")
        return
    
    # 查找所有 session 目录
    session_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("session_")]
    
    if not session_dirs:
        print(f"错误: 在 {logs_dir} 中找不到任何 session 目录")
        return
    
    print("第一夜死亡情况汇总:")
    print("=" * 60)
    print()
    
    results = []
    
    for session_dir in sorted(session_dirs):
        game_file = session_dir / "game_complete.json"
        
        if not game_file.exists():
            continue
        
        try:
            with open(game_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session_id = data.get("session_id", "unknown")
            session_name = session_dir.name
            rounds = data.get("rounds", [])
            
            if not rounds:
                results.append({
                    "session_id": session_id,
                    "session_name": session_name,
                    "eliminated": None,
                    "protected": None,
                    "status": "no_rounds"
                })
                continue
            
            # 第一夜是 rounds[0]
            eliminated = rounds[0].get("eliminated")
            protected = rounds[0].get("protected")
            
            results.append({
                "session_id": session_id,
                "session_name": session_name,
                "eliminated": eliminated,
                "protected": protected,
                "status": "ok"
            })
            
        except Exception as e:
            print(f"读取 {game_file} 时出错: {e}")
            continue
    
    # 显示结果
    for result in results:
        session_id = result["session_id"]
        session_name = result["session_name"]
        eliminated = result["eliminated"]
        protected = result["protected"]
        
        if result["status"] == "no_rounds":
            print(f"Session {session_id} ({session_name}): 无游戏数据")
        elif eliminated:
            print(f"Session {session_id} ({session_name}): {eliminated} 死亡 (被保护: {protected})")
        else:
            print(f"Session {session_id} ({session_name}): 无人死亡 (被保护: {protected})")
    
    print()
    print("=" * 60)
    print(f"总计: {len(results)} 个游戏")
    
    # 统计
    deaths = [r for r in results if r.get("eliminated")]
    no_deaths = [r for r in results if r.get("status") == "ok" and not r.get("eliminated")]
    
    print(f"第一夜有死亡: {len(deaths)} 个")
    print(f"第一夜无死亡: {len(no_deaths)} 个")
    
    if deaths:
        print("\n第一夜死亡的玩家:")
        death_counts = {}
        for r in deaths:
            player = r["eliminated"]
            death_counts[player] = death_counts.get(player, 0) + 1
        
        for player, count in sorted(death_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {player}: {count} 次")

if __name__ == "__main__":
    check_first_night_deaths()
