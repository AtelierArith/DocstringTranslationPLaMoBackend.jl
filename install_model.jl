import REPL
import Markdown
import CondaPkg

lang = "English"

CondaPkg.withenv() do
	run(`plamo-translate --to $(lang) --input "plamo-translate のモデルのインストールが完了しました"`)
	run(`plamo-translate --to $(lang) --input "この Julia パッケージは日本産の LLM [PLaMo](https://plamo.preferredai.jp/) をバックエンドとする Julia のマニュアルを翻訳する機能を提供します．"`)
	run(`plamo-translate --to $(lang) --input "このパッケージは CLI ツール [plamo-translate-cli](https://github.com/pfnet/plamo-translate-cli) をラップすることで翻訳機能を実現しています．"`)
end
