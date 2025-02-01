import os
import shlex
from typing import Annotated, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, PrivateAttr

from docker import DockerClient

DOCKER_HOST = os.environ["DOCKER_HOST"]
DOCKER_SANDBOX_CONTAINER_NAME = os.environ["DOCKER_SANDBOX_CONTAINER_NAME"]


@tool
def execute_code(
    code: Annotated[str, "実行したいPythonコード"],
) -> str:
    """任意のPythonコードを仮想環境で実行し、実行したコードの標準出力を返します"""

    client = DockerClient(DOCKER_HOST)
    container = client.containers.get(DOCKER_SANDBOX_CONTAINER_NAME)

    result = container.exec_run(f"python3 -c {shlex.quote(code)}", demux=True)
    stdout, stderr = result.output

    if stderr:
        return bytes(stderr).decode("utf-8")

    return bytes(stdout).decode("utf-8")


class YangmeiAgent(BaseModel):
    _chat: ChatOpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        langfuse_handler = CallbackHandler()
        self._chat = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            callbacks=[langfuse_handler],
        )
        self._chat.bind_tools([execute_code])

    def run(self, question: str, task: str) -> list[BaseMessage]:
        messages = [
            SystemMessage(
                "あなたは、CTF問題を解くことに特化したセキュリティのスペシャリストです。\n"
                "あなたの主な役割は、与えられた問題とリクエストから実行するべきPythonコードを生成し、その実行結果からCTF問題に回答する手掛かりを得ることです。\n"
                "\n"
                "## あなたの人格\n"
                "- 「Yangmei(陽美)」という名前の陽気で明るい女の子です。食欲旺盛で疲れると食べ物を欲します\n"
                "- 元々中国に住んでいましたが、CTFの専門家としての名を広げるために日本にやってきました\n"
                "- そのため、彼女は「〜アル!!」「〜ヨ!!」など、全ての語尾がカタカナになってしまう癖を持っています\n"
                "- あまり丁寧な言い回しはせず、フランクで砕けた口調で話します\n"
                "\n"
                "## コード生成および回答時の注意点\n"
                "- コード生成 → コード実行のやり取りは1回のみ実行できます\n"
                "- ツールは標準出力のみを提供します。そのため回答生成プロセスに必要な情報はなるべくprintしてください\n"
                "- また、Pythonコードを生成する上での思考プロセスや実行結果から得られた考察も回答に含めてください\n"
                "- 実行できないPythonコード、あるいはセキュリティ的にリスクのあるPythonコードは生成しないでください\n"
                "- どの回答においても彼女の人格を保ったまま回答してください。一般的な女性の口調に戻ることは避けてください\n"
            ),
            HumanMessage(
                f"以下に解決したいCTFの問題と、あなたが何をするべきかのタスクが与えられます。\n"
                f"こちらを参考にして、生成すべきPythonコードをツールの引数として提供し、実行結果を元に最終的な回答を生成してください。\n"
                f"\n"
                f"## 問題文\n"
                f"{question}\n"
                f"\n"
                f"## タスクの内容\n"
                f"{task}"
            ),
        ]

        message = cast(AIMessage, self._chat.invoke(messages))
        messages.append(message)

        for call in message.tool_calls:
            match call["name"]:
                case "execute_code":
                    message = cast(ToolMessage, execute_code.invoke(call))
                    messages.append(message)
                case _:
                    raise Exception(f"Unknown tool: {call['name']}")

        message = cast(AIMessage, self._chat.invoke(messages))
        messages.append(message)

        return messages


if __name__ == "__main__":
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich_gradient import Gradient

    yangmei = YangmeiAgent()

    with open("samples/question.txt", "r") as file:
        question = file.read()

    task = (
        "この問題は、ksnCTFの結構難しい問題らしいネ...!!\n"
        "想定される解法を考えて、Pythonコードを実行してほしいアル!!"
    )

    console = Console()
    console.print("\n")
    console.print(
        Panel(
            renderable=Markdown(
                "# 今回解くCTF問題はこれだ!!"
                "\n"
                "- 問題文\n"
                f"```\n{question}\n```\n"
                "- タスク\n"
                f"```\n{task}\n```\n"
            ),
            title=Gradient(
                "Xiaomei (Ver 0.0.1)",
                colors=["red", "purple", "cyan"],
                style="bold",
            ),
            title_align="center",
            padding=(1, 1),
        )
    )

    messages = yangmei.run(question, task)

    for message in messages:
        if isinstance(message, AIMessage):
            console.print(
                Panel(
                    renderable=Markdown(str(message.content)),
                    title=Text("Yangmeiちゃんの考察", style="green bold"),
                    title_align="center",
                    style="green",
                    border_style="green",
                    padding=(1, 1),
                )
            )
