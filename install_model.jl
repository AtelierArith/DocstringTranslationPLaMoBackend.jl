import REPL
import Markdown
import CondaPkg

lang = "English"

CondaPkg.withenv() do
	run(`plamo-translate --to $(lang) --input "こんにちは. plamo-translate のモデルのインストールが完了しました"`)
	run(`plamo-translate --to $(lang) --input "[PLaMo](https://plamo.preferredai.jp/) という国産 LLM をバックエンドとするJuliaのマニュアル翻訳パッケージを開発しました．
このパッケージは CLI ツール [plamo-translate-cli](https://github.com/pfnet/plamo-translate-cli) をラップすることで翻訳機能を実現しています．
"`)
end
