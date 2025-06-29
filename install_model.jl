import REPL
import Markdown
import CondaPkg

lang = "English"

CondaPkg.withenv() do
	run(`plamo-translate --to $(lang) --input "こんにちは. plamo-translate のモデルのインストールが完了しました"`)
end
