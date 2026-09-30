#!/bin/zsh
# Local compile test on the MacBook: FUTO Keyboard <tag> + patches/*.patch -> debug-signed release APK.
# Usage: tools/mac-build.sh [upstream tag]   (default: latest stable)
set -eu   # no pipefail: "yes | sdkmanager" ends with yes killed by SIGPIPE
export PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/bin:$PATH
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=$HOME/Library/Android/sdk
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=$HOME/futo-build
TAG=${1:-$(curl -s https://api.github.com/repos/futo-org/android-keyboard/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')}

if [ ! -d "$ANDROID_HOME/platform-tools" ]; then
  mkdir -p "$ANDROID_HOME"
  yes | /opt/homebrew/share/android-commandlinetools/cmdline-tools/latest/bin/sdkmanager --sdk_root="$ANDROID_HOME" --licenses > /dev/null
fi

mkdir -p "$WORK"; cd "$WORK"
if [ ! -d src ]; then
  git clone --filter=blob:none https://github.com/futo-org/android-keyboard.git src
fi
cd src
git fetch --tags -q
git reset -q --hard; git clean -qfdx -e build -e .gradle
git checkout -q "$TAG"
git submodule update --init --recursive --depth 1
NDK=$(grep -oE "ndkVersion '[^']+" build.gradle | cut -d"'" -f2)
yes | /opt/homebrew/share/android-commandlinetools/cmdline-tools/latest/bin/sdkmanager --sdk_root="$ANDROID_HOME" \
  "platform-tools" "ndk;$NDK" "cmake;3.22.1" > /dev/null
echo "sdk.dir=$ANDROID_HOME" > local.properties

for p in "$REPO"/patches/*.patch; do echo "== $p"; git apply --3way --whitespace=nowarn "$p"; done
# on-device cleanup runtime: pinned modern llama.cpp for native/llmcleanup (see patches/0003)
LLAMA_TAG=$(cat "$REPO/patches/llama.cpp.version")
rm -rf native/llmcleanup/llama.cpp
git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp native/llmcleanup/llama.cpp
perl -0pi -e 's/(productFlavors \{.*?\n(\s*)stable \{\n)/$1$2    applicationIdSuffix ".greek"\n/s' build.gradle
perl -0pi -e 's/(\n\s*stable \{.*?buildConfigField "boolean", "UPDATE_CHECKING", )"true"/$1"false"/s' build.gradle
grep -rl --include='*.xml' 'name="english_ime_name"' java/res translations 2>/dev/null | \
  xargs sed -i '' -E 's#(name="english_ime_name"[^>]*>)[^<]*<#\1FUTO Keyboard (Greek patch)<#'

VERSION_NAME="$TAG-greek.local" VERSION_CODE=$(( $(git rev-list --first-parent --count HEAD) * 100 + 99 )) \
  ./gradlew assembleStableRelease -s --no-daemon
ls -la build/outputs/apk/stable/release/
