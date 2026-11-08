# DocstringTranslationPLaMoBackend.jl

Julia の docstring と Documenter ページを、同梱した **PLaMo.jl のネイティブ Julia 実装**で翻訳します。Python・Conda・`plamo-translate` CLI は不要です。

PLaMo の実装は `src/PLaMo/` に同梱しているため、インストール時に非公開の upstream リポジトリへアクセスする必要はありません。モデルのアセットは Scratch.jl でキャッシュされ、遅延ロードされた 1 つのモデルをすべての翻訳で使い回します。

## セットアップ

Julia **1.13 以降**が必要です。

```sh
git clone https://github.com/AtelierArith/DocstringTranslationPLaMoBackend.jl.git
cd DocstringTranslationPLaMoBackend.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.build("DocstringTranslationPLaMoBackend")'
julia --threads=8 --project
```

```julia
using DocstringTranslationPLaMoBackend

@switchlang! :Japanese # :ja や "ja" も使えます
@doc sin

@switchlang! :English # :en や "en" も使えます
```

`@switchlang!` は、元のパッケージと同様に、現在の Julia プロセスの `Docs.parsedoc` と `Documenter.Page` を切り替えます。他のパッケージのプリコンパイル中ではなく、REPL やドキュメントのビルドスクリプトから呼び出してください。パース済みの docstring も言語を切り替えられ、元の Markdown も引き続き参照できます。Julia のコードブロック（フェンス付き）は翻訳に送られません。

## モデルキャッシュ: Scratch.jl

```julia
using DocstringTranslationPLaMoBackend

install_model!()
DocstringTranslationPLaMoBackend.model_directory()
```

`Pkg.build("DocstringTranslationPLaMoBackend")` を実行すると、モデルが自動でインストールされます。
パッケージの初回インストール時や instantiate 時にも、Julia がこのビルドを実行することがあります。
インストーラーは、モデルリビジョン `cc8e88169807d5a272e8ff06db12eb0129b67f81` に固定した **PLaMo 3 Translate 0.8B** を、Julia の標準ライブラリ `Downloads` でダウンロードします。各アセットは記録済みの SHA-256 と照合してから、アトミックに置き換えられます。2 回目以降のインストールでは、正しいファイルは再利用し、欠けているファイルや壊れているファイルだけを修復します。

モデルのディレクトリは、このパッケージの UUID の下で Scratch.jl が管理します。デフォルトの depot では次のような場所になります。

```text
~/.julia/scratchspaces/691b352d-20d8-448b-9a22-61c32af05849/plamo-3-translate-0.8b-<revision>/
```

この場所は Julia の depot 設定に従います。モデルの重みは、このリポジトリ・パッケージのインストール先・作業ディレクトリのいずれにも保存されません。パッケージを import すると Scratch ディレクトリは作成されますが、モデルのダウンロードやロードは行われません。アセットはビルド時にインストールされ、モデル本体は最初の翻訳時に遅延ロードされます。キャッシュのインストールや修復は、`install_model!()` を明示的に実行しても行えます。

手元にある公式チェックポイントを、再ダウンロードせずに Scratch キャッシュへコピーするには次のようにします。

```julia
install_model!(; source_directory="/path/to/plamo-3-translate-0.8b")
```

自動ビルドで既存のチェックポイントを使うには、instantiate またはビルドの前に `PLAMO_SOURCE_DIR` を設定します。

```sh
PLAMO_SOURCE_DIR=/path/to/plamo-3-translate-0.8b \
  julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("DocstringTranslationPLaMoBackend")'
```

取り込むファイルは、2 つのライセンスファイルも含めて、固定した公式リビジョンと一致している必要があります。ファイルは検証してからコピーされ、外部のディレクトリが実行時のモデルの場所になることはありません。

ダウンロードされるアセットは約 **1.78 GB** です。デフォルトの BF16 実装では、SSSE3 対応の x86 CPU で約 **2.73 GB の重みデータ**を保持します。これに加えて、正規化パラメータ、KV キャッシュ、作業領域、アロケーターのオーバーヘッド、ロード時の一時領域が必要です。元の BF16 値をそのまま保持して F32 で演算し、辞書圧縮は可逆です。その他の CPU アーキテクチャでは移植性の高い実装を使いますが、実際の推論を検証したのは Intel CPU だけです。

重みには、このパッケージの MIT ライセンスとは別に [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en) が適用されます。インストーラーは英語版と日本語版のライセンスファイル原本もキャッシュします。

## Apple Silicon GPU

Apple Silicon の Mac では、デフォルトで GPU と Q4_0 重みを使って翻訳します。追加のパッケージや `using Metal` は不要です。

```julia
using DocstringTranslationPLaMoBackend
@switchlang! :Japanese
```

Metal.jl は通常の依存パッケージですが、ロードされるのは Apple Silicon（`aarch64` の macOS）の場合だけです。その他のプラットフォームでは、インストールはされてもロードはされません。デフォルトの `device=:auto` は、対応する GPU が使える場合は Metal を、そうでない場合は CPU を選びます。CPU を強制するには `device=:cpu` を指定します。`:metal` を明示した場合、Metal が使えなければエラーになります。GPU へのリクエストと Markdown の走査は Julia のスレッド並列なしで実行されるため、Metal では `JULIA_NUM_THREADS=8` は不要です。
デコード用カーネルは、`Metal.KernelException` の報告は保ちつつ、デフォルトでは GPU 例外の詳細情報を省略します。デバイスの診断情報をすべて残すには、Julia を `-g2` 付きで起動してください。この設定によって fast math が有効になったり、推論精度が変わったりすることはありません。

`weight_storage=:bf16` のとき、GPU 経路は固定した BF16 重みをそのまま保持します。どの形式でも、活性値と累積計算は Float32 です。プロンプト処理を速くするため、`prefill_storage=:auto` は、GPU とシステムの空きメモリがどちらも「8 GiB + 設定した KV キャッシュ容量」を超える場合に、Float32 の射影重みも追加でキャッシュします。このヒューリスティックはロードや一時バッファの余裕を見込んでいますが、将来の空きメモリを保証するものではありません。確保に失敗した場合、自動選択は BF16 にフォールバックします。
このキャッシュで GPU 上の確保量が約 2.2 GB 増えます。追加のキャッシュを避けるには `configure_plamo!(; device=:metal, prefill_storage=:bf16)` を、明示的に有効にするには `:f32` を指定してください。`weight_storage=:f32` の場合は射影重みがすでにすべて Float32 なので、追加のコピーは作られません。

浮動小数点演算の順序の違いにより、結果が CPU とわずかに異なることがあります。GPU 経路と CPU 経路は、それぞれモデルとリクエストごとの KV キャッシュを選択したデバイス上に保持します。設定を変更すると、以前のモデルは解放されます。

## 推論の設定

```julia
configure_plamo!(;
    device=:auto,       # :cpu または :metal を明示的に選べます
    prefill_storage=:auto, # Metal のプロンプト用キャッシュ: :bf16 または :f32
    context_size=4096,
    max_tokens=1024,
    weight_storage=:auto, # Metal では :q4_0、CPU では :bf16。:f32 はメモリを多く使います
)
```

Metal はブロック量子化に対応しており、Metal でのデフォルトは `:q4_0` です。

```julia
DocstringTranslationPLaMoBackend.configure_plamo!(
    device=:metal,
    weight_storage=:q8_0, # 元の重みを保つなら :bf16
)
```

これらのモードは、翻訳器のロード時に固定の BF16 チェックポイントを量子化します。追加のモデルダウンロードや GGUF 変換は不要です。量子化値は Q8_0/Q4_0 の参照実装と同じ丸めで計算し、GPU 用の独自レイアウトで保持します。量子化によって翻訳結果が変わることがあります。活性値・KV キャッシュ・累積計算は Float32 のままです。量子化した重みは Metal 専用で、CPU では使えません。

M2 Max・Julia 1 スレッドでの最適化済み Metal デコードは、BF16 で約 190 token/s、Q8_0 で 246、Q4_0 で 404（Float32 は 92）でした。Q4_0 での短い翻訳は約 30 ms でした。同じトークン列を交互に実行した別の比較では、デフォルトの Float16 KV と Flash Attention 自動選択を使った llama.cpp が約 422 token/s、Metal.jl が 405 でした。llama.cpp を上回るという目標はまだ達成できていません（約 4% のスループット向上が必要です）。
いずれもウォーム状態の計測で、モデルのロードとコンパイルは含みません。詳しくは[ベンチマークの詳細と出力の違い](benchmark/README.md#register-resident-dual-rms-normalization)を参照してください。

展開済みの prefill 用キャッシュには、選択した形式を逆量子化した値が入ります。`prefill_storage=:native` を指定するとこのキャッシュを無効にでき、prefill 中に射影を 1 つずつ展開します。`:bf16` は、このキャッシュを無効にする指定の別名として残しています。
量子化は翻訳器を作るたびにやり直され、別の Scratch アセットとして保存されることはありません。ロード時間・メモリ・翻訳結果のトレードオフについては、ベンチマークの比較を参照してください。

### Q4_0 について

Q4_0 は llama.cpp（GGML）で使われる 4bit のブロック量子化です。このパッケージは同じ丸め方を自前で実装しています（[`src/MetalBackend/quantization.jl`](src/MetalBackend/quantization.jl)）。

**仕組み**

1. 重み行列を 32 個ずつのブロックに区切ります。
2. ブロックごとに、絶対値が最大の値 `max`（符号付き）からスケール `d = max / -8` を求め、Float16 で保存します。
3. 各重みを `q = clamp(trunc(w/d + 8.5), 0, 15)` で 0〜15 の 4bit 整数にし、2 つずつ 1 バイトに詰めます。
4. 推論時は `w ≈ (q - 8) * d` で元の値に戻します。

元の値の 16 段階近似なので、BF16 より精度は落ちます。

**サイズ**

32 個の重みが、データ 16 バイトとスケール 2 バイトの計 18 バイトになります。1 重みあたり 4.5bit です（BF16 は 16bit）。重みデータは約 3.6 分の 1 になり、GPU 上の確保量は 3.75 GiB から 2.55 GiB に減りました。prefill 用の Float32 キャッシュなどを含めた値なので、比率は 3.6 倍より小さくなります。

**速くなる理由**

- decode（1 トークンずつの生成）: 行列ベクトル積が中心で、計算よりも重みをメモリから読む速度が律速になります。読むバイト数が減る分だけ速くなり、M2 Max では BF16 の 190 token/s に対して 404 token/s でした。
- prefill（プロンプト処理）: 計算が律速で、展開済みの Float32 キャッシュを使うため、どの形式でも約 5,000 token/s で差は出ません。

**この実装の特徴**

- 量子化するのは重みだけで、活性値・KV キャッシュ・累積計算は Float32 のままです。
- 量子化は翻訳器を作るたびに CPU で実行し、保存はしません。そのためセットアップに約 5 s かかります（BF16 は約 2.8 s）。
- 値は GGML と同じですが、GPU 向けにスケールを分けて持つ独自のレイアウトなので、GGUF ファイルとの互換性はありません。

**品質**

8 例中 6 例で BF16 と同じ訳文になり、残り 2 例は言い回しが変わりました。正確さを重視するなら `:q8_0`（8bit で、8 例すべて BF16 と一致、246 token/s）か `:bf16` が向いています。

### その他の設定

設定を変更するとキャッシュ済みの翻訳器は解放され、次の翻訳時に再ロードされます。`context_size` は、プロンプトと `max_tokens` の合計を収められる大きさにしてください。不完全な翻訳や空の翻訳はエラーになり、完成したドキュメントとしては保存されません。長い段落や docstring を扱う場合は、たとえば `context_size=8192, max_tokens=2048` のように両方の上限を引き上げてください。

ネイティブのリクエストは 1 つのモデルを共有し、直列に処理されます。CPU では、Markdown の走査と PLaMo の射影カーネルが Julia のスレッドを使えます。Metal では走査は逐次的に行われます。CPU のチューニングとして、別途 `LinearAlgebra.BLAS.set_num_threads(1)` を設定することもできます。このパッケージがグローバルな BLAS 設定を変えることはありません。

## 翻訳キャッシュと Documenter

翻訳結果も Scratch.jl の別キャッシュ `translation-plamo3-v3-<model-revision>` に保存されます。エントリーは、元の Markdown・パッケージとバージョン・正規化した言語コードをキーにしています。リビジョンとスキーマで名前空間を分けているため、以前の Python 版や古い走査方式の結果が新しいモデルの結果と混ざることはありません。古いキャッシュは削除されません。Documenter の絶対パスは、そのままキャッシュの保存先になることはなく、Scratch 内に収められます。

```julia
using DocstringTranslationPLaMoBackend
import Documenter

DocstringTranslationPLaMoBackend.switchtargetpackage!(MyPackage)
@switchlang! :Japanese
Documenter.makedocs(; modules=[MyPackage])
```

見出しと表の文章は、見出しレベル・表の配置・コードだけのセルを保ったまま翻訳されます。見出しの表示テキストは Documenter の元のセクション ID（明示的な `@id` ラベルも含む）を保持するため、ページ内やページ間の参照は引き続き解決されます。コードブロックは変更されません。既存の Markdown リンクの保護と、その際に元の段落へ戻すフォールバックも維持しています。生成された翻訳は、ドキュメントを公開する前に確認が必要です。

## テストと実装の出自

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

通常のテストは、モデルのダウンロードも Python の起動もせずに実行できます。対象は、ソースの完全性、Scratch の設定、チェックサム検証、アトミックなインストール、オフラインでの再利用、言語の別名、不完全な出力のエラー、翻訳キャッシュ、docstring と Documenter のフックです。実モデルを使うオプションのチェックも実行するには次のようにします。

```sh
PLAMO_SOURCE_DIR=/path/to/plamo-3-translate-0.8b \
  JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  julia --project -e 'using Pkg; Pkg.test()'
```

Apple Silicon では、Metal 固有のテストは別の環境で実行します。

```sh
julia --project=test/metal -e 'using Pkg; Pkg.instantiate()'
julia --project=test/metal test/metal.jl
PLAMO_METAL_REAL_MODEL=1 julia --project=test/metal test/metal.jl
# オプション: 展開した Float32 の CPU/GPU モデルは、かなり多くのメモリを必要とします。
PLAMO_METAL_F32_MODEL=1 julia --project=test/metal test/metal.jl
# 2048 トークンをキャッシュした後のデコードと、長いコンテキスト用の attention 経路を確認します。
PLAMO_METAL_LONG_MODEL=1 julia --project=test/metal test/metal.jl
# Q8_0/Q4_0 を独立した Float32 の CPU モデルと比較します（追加のメモリが必要です）。
PLAMO_METAL_QUANT_MODEL=1 JULIA_NUM_THREADS=1 julia --project=test/metal test/metal.jl
```

`PLAMO_METAL_REAL_MODEL=1` は、実モデルでの CPU/GPU のロジット、prefill のチャンク境界、エンドツーエンドの翻訳も確認します。キャッシュ済みのモデルが必要です。CI は、macOS ランナーで使える GPU がある場合に Metal カーネルのテストを実行し、ない場合は警告を出します。実モデルのチェックは、明示的に実行するローカル専用のテストです。

ウォーム状態の推論ベンチマーク、GPU プロファイリング、AllocCheck によるアロケーション診断、llama.cpp との比較結果については [benchmark/README.md](benchmark/README.md) を参照してください。M2 Max では、記載したウォーム状態のワークロードにおいて、Metal で `JULIA_NUM_THREADS=1` を使い、llama.cpp と同程度のスループットに達しました。

ネイティブ実装のソースは、PLaMo.jl のコミット `e919a8a13e18aa997ee0d7708cd35fdb00812fdc` から `src/PLaMo/` にコピーしています（`PLaMo.jl` の `include` パスのみ本リポジトリの配置に合わせて変更）。[出自と更新手順](src/PLaMo/UPSTREAM.md)、[ソースのチェックサム](src/PLaMo/checksums.toml)、[upstream の MIT ライセンス](src/PLaMo/LICENSE)を参照してください。実行時のセットアップには、登録済みの Julia パッケージと標準ライブラリだけを使います。
