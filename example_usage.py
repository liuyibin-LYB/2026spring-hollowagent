"""
示例：如何使用 TreeholeRAGAgent 的四种模式。
"""

from agent import TreeholeRAGAgent


def example_quick_qa():
    print("\n" + "=" * 60)
    print("示例 1: 日常 Q&A")
    print("=" * 60)

    agent = TreeholeRAGAgent()
    result = agent.mode_quick_qa("我想选一些 AI 和机器学习相关课程，先给我一个入门路线。")
    print(f"\n【搜索次数】{result['search_count']}")
    print(f"【会话ID】{result['session_id']}")
    print(f"【参考来源】{result['num_sources']} 个帖子")


def example_deep_research():
    print("\n" + "=" * 60)
    print("示例 2: Deep Research")
    print("=" * 60)

    agent = TreeholeRAGAgent()
    result = agent.mode_deep_research("帮我系统研究一下北大树洞里对 AI 引论这门课的整体评价。")
    print(f"\n【搜索次数】{result['search_count']}")
    print(f"【会话ID】{result['session_id']}")
    print(f"【参考来源】{result['num_sources']} 个帖子")


def example_daily_digest():
    print("\n" + "=" * 60)
    print("示例 3: 每日神帖汇总")
    print("=" * 60)

    agent = TreeholeRAGAgent()
    result = agent.mode_daily_hot_digest(recent_post_count=40)
    print(f"\n【参考来源】{result['num_sources']} 个帖子")
    print("【产物】")
    for key, value in result.get("artifacts", {}).items():
        if value:
            print(f"  - {key}: {value}")


def example_thorough_search():
    print("\n" + "=" * 60)
    print("示例 4: Thorough Search")
    print("=" * 60)

    agent = TreeholeRAGAgent()
    result = agent.mode_thorough_search(
        keywords=["AI 引论", "人工智能引论", "爱引论"],
        question="请总结大家最常提到的优点、缺点和 workload。",
    )
    print(f"\n【语料规模】{result['num_sources']} 个帖子")
    print("【产物】")
    for key, value in result.get("artifacts", {}).items():
        if value:
            print(f"  - {key}: {value}")


def main():
    print("PKU Treehole RAG Agent - 编程调用示例")
    print("=" * 60)
    print("\n可运行示例：")
    print("  1 - 日常 Q&A")
    print("  2 - Deep Research")
    print("  3 - 每日神帖汇总")
    print("  4 - Thorough Search")
    print("  a - 全部运行")
    print("  q - 退出")

    choice = input("\n请选择 (1-4/a/q): ").strip().lower()
    if choice == "q":
        return
    if choice == "1":
        example_quick_qa()
    elif choice == "2":
        example_deep_research()
    elif choice == "3":
        example_daily_digest()
    elif choice == "4":
        example_thorough_search()
    elif choice == "a":
        example_quick_qa()
        example_deep_research()
        example_daily_digest()
        example_thorough_search()
    else:
        print("无效选择")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
