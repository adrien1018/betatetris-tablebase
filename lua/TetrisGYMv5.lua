currentSeedAddress = 0x0037

gameStateAddress = 0x00C0
playStateAddress = 0x0048
isLevelEnterAddress = 0x0764
currentLevelAddress = 0x0765
nowTetriminoIDAddress = 0x0042
nextTetriminoIDAddress = 0x00BF

tetriminoXAddress = 0x0041
tetriminoYAddress = 0x0040
tetriminoRotateAddress = 0x0042

-- use O(n) queue anyway for simplicity
recvQueue = ""
sendQueue = ""

function trySend(tcp, msg)
  msg = msg or ""
  if string.len(msg) ~= 0 then
    sendQueue = sendQueue .. msg
  end
  if string.len(sendQueue) == 0 then
    return
  end
  local x, y, z = tcp:send(sendQueue)
  if x == nil then
    x = z
  end
  sendQueue = string.sub(sendQueue, x + 1)
end

function tryReceive(tcp, size)
  sizeToRead = math.max(0, size - string.len(recvQueue))
  if sizeToRead == 0 then
    local ret = string.sub(recvQueue, 1, size)
    recvQueue = string.sub(recvQueue, size + 1)
    return ret
  end
  local x, y, z = tcp:receive(sizeToRead)
  if x == nil then
    x = z
  end
  recvQueue = recvQueue .. x
  if string.len(recvQueue) >= size then
    local ret = string.sub(recvQueue, 1, size)
    recvQueue = string.sub(recvQueue, size + 1)
    return ret
  else
    return nil
  end
end

function resetQueue(tcp)
  tcp:receive(1000000)
  sendQueue = ""
  recvQueue = ""
end

--[[
TCP stream format:
- Piece (1 byte): (0x00~0x06)
- Starting level (1 byte): (0x12 or 0x13)
- Piece position (3 bytes): [rotate](0x00~0x03) [x](0x00~0x13) [y](0x00~0x09)
- Move sequence (len+2 bytes): 0xfe [seq length] [seq...]
  - Each byte is a frame or'ed by following keys:
    - 0x01 (left)
    - 0x02 (right)
    - 0x04 (A)
    - 0x08 (B)
- Procedure
  - Game start (sent 4 bytes): 0xff [current piece] [next piece] [starting level]
  - Game loop
    - Current piece microadjustment sequence (appended to next piece move sequence) (receive)
    - Next piece move sequence (receive)
    - Locked position + next piece (sent 5 bytes): 0xfd [locked position] [next piece]
--]]

pieceMap = {}
pieceMap[2] = 0
pieceMap[7] = 1
pieceMap[8] = 2
pieceMap[10] = 3
pieceMap[11] = 4
pieceMap[14] = 5
pieceMap[18] = 6
--             T            J         Z    O     S         L         I
rotateMap = {3, 0, 1,  1, 2, 3, 0,  0, 1,  0,  0, 1,  3, 0, 1, 2,  1, 0}
rotateMap[0] = 2

function sendStartGame(tcp, level)
  local currentPiece = memory.readbyteunsigned(nowTetriminoIDAddress)
  local nextPiece = memory.readbyteunsigned(nextTetriminoIDAddress)
  local msg = string.char(0xff, pieceMap[currentPiece], pieceMap[nextPiece], level)
  print('startGame', currentPiece, nextPiece, level)
  trySend(tcp, msg)
end

function printBytes(bytes)
  local str = ''
  for i = 1,string.len(bytes) do
    str = str .. tostring(string.byte(bytes, i)) .. ','
  end
  if string.len(str) > 0 then
    str = string.sub(str, 1, string.len(str) - 1)
  end
  print('[' .. str .. ']')
end

function receiveSequence(tcp, seq)
  if not seq.length then
    local p = tryReceive(tcp, 2)
    if p then
      if string.byte(p, 1) == 0xfe then
        seq.length = string.byte(p, 2)
      end
    end
  end
  if seq.length and not seq[seq.length] then
    local p = tryReceive(tcp, seq.length)
    if p then
      for i = 1,seq.length do
        local x = string.byte(p, i)
        local buttons = {}
        if x % 2 >= 1 then buttons.left = true end
        if x % 4 >= 2 then buttons.right = true end
        if x % 8 >= 4 then buttons.A = true end
        if x % 16 >= 8 then buttons.B = true end
        seq[i] = buttons
      end
      return true
    end
  end
  return false
end

function receiveTwoSequence(tcp, curSeq, nextSeq, nFrame, block)
  if nextSeq.length and nextSeq[nextSeq.length] then
    return
  end
  if not (curSeq.length and curSeq[curSeq.length]) then
    if block == 1 then
      tcp:settimeout(nil, 't')
      for i = 1,5 do
        if receiveSequence(tcp, curSeq) then break end
      end
      tcp:settimeout(0.008, 't')
    else
      receiveSequence(tcp, curSeq)
    end
  end
  if curSeq.length and curSeq[curSeq.length] then
    if block >= 1 then
      tcp:settimeout(nil, 't')
      for i = 1,5 do
        if receiveSequence(tcp, nextSeq) then break end
      end
      tcp:settimeout(0.008, 't')
    else
      receiveSequence(tcp, nextSeq)
    end
  end
end

function getLines()
  local r1 = memory.readbyteunsigned(0x0051)
  local r2 = memory.readbyteunsigned(0x0050)
  local lns = r1
  lns = lns * 10 + math.floor(r2 / 16)
  lns = lns * 10 + r2 % 16
  return lns
end

function getScore()
  local r1 = memory.readwordunsigned(0x0008)
  local r2 = memory.readwordunsigned(0x000A)
  return r2 * 65536 + r1
end

function gameLoop(tcp, level, seed, terminate)
  resetQueue(tcp)
  sendStartGame(tcp, level)
  local endGame = false
  local nextSequence = {length=1}
  nextSequence[1] = {}
  while not endGame do
    local curSequence = {}
    local fNextSequence = {}
    local inMicro = false
    local currentFrame = 1
    local nFrame = 0
    local st = memory.readbyteunsigned(playStateAddress)
    while st == 1 do
      trySend()
      local block = 0
      if inMicro and currentFrame == 1 then
        block = 1
      end
      receiveTwoSequence(tcp, curSequence, fNextSequence, nFrame, block)
      if inMicro then
        if curSequence[currentFrame] then
          joypad.set(1, curSequence[currentFrame])
          currentFrame = currentFrame + 1
        end
      else
        if nextSequence[currentFrame] then
          joypad.set(1, nextSequence[currentFrame])
          currentFrame = currentFrame + 1
          if currentFrame > nextSequence.length then
            inMicro = true
            currentFrame = 1
          end
        end
      end
      emu.frameadvance()
      st = memory.readbyteunsigned(playStateAddress)
      nFrame = nFrame + 1
    end
    --print('total frames:', nFrame)
    --print(getScore(), getLines())
    local rotate = rotateMap[memory.readbyteunsigned(tetriminoRotateAddress)]
    local x = memory.readbyteunsigned(tetriminoXAddress)
    local y = memory.readbyteunsigned(tetriminoYAddress)
    if false then -- early terminate
      while st ~= 10 do
        emu.frameadvance()
        st = memory.readbyteunsigned(playStateAddress)
      end
    end
    while st ~= 1 do
      if st == 10 then
        endGame = true
        print(getScore(), getLines())
        break
      end
      emu.frameadvance()
      st = memory.readbyteunsigned(playStateAddress)
    end
    receiveTwoSequence(tcp, curSequence, fNextSequence, nFrame, 2)
    nextSequence = fNextSequence
    local nextPiece = memory.readbyteunsigned(nextTetriminoIDAddress)
    trySend(tcp, string.char(0xfd, rotate, x, y, pieceMap[nextPiece]))
  end
  if not terminate then return end
  for i = 1,60 do
    emu.frameadvance()
  end
  while memory.readbyteunsigned(gameStateAddress) == 4 do
    joypad.set(1, {start=true})
    for i = 1,5 do
      emu.frameadvance()
    end
  end
end

--- menu

function waitStart()
  for i = 1,14 do
    emu.frameadvance()
  end
end

function enableDoubleKs()
  for i = 1,4 do
    emu.frameadvance()
    joypad.set(1, {up=true})
    emu.frameadvance()
  end
  joypad.set(1, {right=true})
  emu.frameadvance()
  for i = 1,4 do
    emu.frameadvance()
    joypad.set(1, {down=true})
    emu.frameadvance()
  end
end

function moveToSeed()
  for i = 1,2 do
    emu.frameadvance()
    joypad.set(1, {down=true})
    emu.frameadvance()
  end
end

function randomSeed()
  joypad.set(1, {select=true})
  emu.frameadvance()
end

function hexToNum(chr)
  local ascii = string.byte(chr)
  if ascii >= 48 and ascii <= 57 then
    return ascii - 48
  end
  if ascii >= 65 and ascii <= 70 then
    return ascii - 65 + 10
  end
  if ascii >= 97 and ascii <= 102 then
    return ascii - 97 + 10
  end
  return 0
end

function currentSeed(digit)
  local addr = currentSeedAddress + math.floor((digit - 1) / 2)
  local val = memory.readbyteunsigned(addr)
  if digit % 2 == 0 then
    return val % 16
  else
    return math.floor(val / 16)
  end
end

function inputSeed(seed)
  for i = 1,6 do
    joypad.set(1, {right=true})
    emu.frameadvance()
    emu.frameadvance()
    emu.frameadvance()
    local digit = hexToNum(string.sub(seed, i, i))
    while true do
      local cur = currentSeed(i)
      if cur == digit then
        break
      end
      if (cur + 16 - digit) % 16 < 8 then
        joypad.set(1, {down=true})
      else
        joypad.set(1, {up=true})
      end
      emu.frameadvance()
      emu.frameadvance()
    end
  end
end

function enterMenu()
  joypad.set(1, {start=true})
  for i = 1,5 do
    emu.frameadvance()
  end
end

function startGame(level)
  while memory.readbyteunsigned(isLevelEnterAddress) ~= 1 do
    joypad.set(1, {right=true})
    emu.frameadvance()
    emu.frameadvance()
  end
  while memory.readbyteunsigned(currentLevelAddress) < level do
    joypad.set(1, {up=true})
    emu.frameadvance()
    emu.frameadvance()
  end
  while memory.readbyteunsigned(currentLevelAddress) > level do
    joypad.set(1, {down=true})
    emu.frameadvance()
    emu.frameadvance()
  end
  joypad.set(1, {start=true})
  emu.frameadvance()
  for i = 1,8 do
    emu.frameadvance()
  end
end

function backToMain()
  emu.frameadvance()
  joypad.set(1, {B=true})
  for i = 1,10 do
    emu.frameadvance()
  end
  joypad.set(1, {right=true})
  for i = 1,2 do
    emu.frameadvance()
  end
end

function generateRandomSeed()
  local s = ''
  for i = 1,6 do
    local num = math.random(1, 16)
    s = s .. string.sub('0123456789abcdef', num, num)
  end
  print(s)
  return s
end

math.randomseed(123)
math.random()
local socket = require("socket")
local tcp = assert(socket.tcp())
local ret, msg = tcp:connect("127.0.0.1", 3456)
if not ret then
  print("Connection failed", msg)
  while true do emu.frameadvance() end
end
tcp:settimeout(0.001, 't')

if true then
  emu.poweron()
  waitStart()
  enableDoubleKs()
  moveToSeed()
end

while true do
  local seed = generateRandomSeed()

  inputSeed(seed)
  enterMenu()

  startGame(18)
  gameLoop(tcp, 18, seed, true)
  backToMain()
end
