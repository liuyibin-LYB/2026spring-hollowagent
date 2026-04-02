"""
示例：如何使用 TreeholeRAGAgent 进行编程调用

这个脚本展示了仅模式2（自动检索）下的编程调用方式。
"""

from agent import TreeholeRAGAgent


def example_auto_search_single_turn():
    """示例1：模式2单轮调用"""
    print("\n" + "=" * 60)
    print("示例 1: 模式2单轮调用")
    print("=" * 60)

    agent = TreeholeRAGAgent()

    result = agent.mode_auto_search(
        user_question="我想选一些AI和机器学习相关的课程，有什么推荐吗？"
    )

    print("\n【问题】我想选一些AI和机器学习相关的课程，有什么推荐吗？")
    print(f"\n【搜索次数】{result['search_count']}")
    print(f"\n【回答】\n{result['answer']}")
    print(f"\n【参考来源】共 {result['num_sources']} 个帖子")


def example_auto_search_multi_turn():
    """示例2：模式2多轮调用"""
    print("\n" + "=" * 60)
    print("示例 2: 模式2多轮调用")
    print("=" * 60)

    agent = TreeholeRAGAgent()

    questions = [
        "我想选一些机器学习课程，给几个方向建议。",
        "如果我数学基础一般，优先哪几门？",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n第{i}轮问题: {question}")
        result = agent.mode_auto_search_multi_turn(question)
        print(f"回答预览: {result['answer'][:120]}...")

    agent.save_conversation()


def example_custom_parameters():
    """示例3：低层接口自定义处理"""
    print("\n" + "=" * 60)
    print("示例 3: 低层接口自定义处理")
    print("=" * 60)

    agent = TreeholeRAGAgent()

    posts = agent.search_treehole(
        keyword="转专业",
        max_results=50,
        use_cache=True,
    )

    print(f"找到 {len(posts)} 个相关帖子")

    from utils import format_posts_batch

    context = format_posts_batch(posts[:5])

    response = agent.call_deepseek(
        user_message=f"基于以下内容，总结一下转专业的难点：\n\n{context}",
        system_message="你是一个北大树洞分析助手",
        temperature=0.5,
    )

    print(f"\n【分析结果】\n{response}")


def example_search_only():
    """示例4：仅搜索，不调用LLM"""
    print("\n" + "=" * 60)
    print("示例 4: 仅搜索树洞内容")
    print("=" * 60)

    agent = TreeholeRAGAgent()

    posts = agent.search_treehole("选课攻略", max_results=20)

    print(f"找到 {len(posts)} 个帖子\n")

    for post in posts[:5]:
        print(f"帖子 #{post.get('pid')}")
        print(f"  内容: {post.get('text', '')[:80]}...")
        print(f"  点赞: {post.get('likenum', 0)} | 回复: {post.get('reply', 0)}")
        print()


def main():
    """运行所有示例"""
    print("PKU Treehole RAG Agent - 编程调用示例")
    print("=" * 60)
    print("\n⚠️  注意：运行这些示例前，请确保：")
    print("  1. 已配置 config_private.py")
    print("  2. 已实现 client.py 中的 search_posts() 函数")
    print("  3. 网络连接正常")
    print("\n选择要运行的示例：")
    print("  1 - 模式2单轮调用")
    print("  2 - 模式2多轮调用")
    print("  3 - 低层接口自定义处理")
    print("  4 - 仅搜索不调用LLM")
    print("  a - 运行所有示例")
    print("  q - 退出")

    choice = input("\n请选择 (1-4/a/q): ").strip().lower()

    if choice == 'q':
        return
    elif choice == '1':
        example_auto_search_single_turn()
    elif choice == '2':
        example_auto_search_multi_turn()
    elif choice == '3':
        example_custom_parameters()
    elif choice == '4':
        example_search_only()
    elif choice == 'a':
        example_auto_search_single_turn()
        example_auto_search_multi_turn()
        example_custom_parameters()
        example_search_only()
    else:
        print("无效选择")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
